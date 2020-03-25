import argparse
import json
import os
import sys
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np

import caffe
import load_dtld as driveu_dataset
import progressbar

# Make sure that caffe is on the python path:
caffe_root = "./"
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, "python"))
os.environ["GLOG_minloglevel"] = "2"


class CaffeDetection:
    def __init__(
        self,
        gpu_id,
        model_def,
        model_weights,
        image_resize_width,
        image_resize_height,
        labelmap_file,
    ):
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()

        self.image_resize_width = image_resize_width
        self.image_resize_height = image_resize_height
        # Load the net in the test phase for inference, and configure input preprocessing.
        self.net = caffe.Net(
            model_def,  # defines the structure of the model
            model_weights,  # contains the trained weights
            caffe.TEST,
        )  # use test mode (e.g., don't perform dropout)
        # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer(
            {"data": self.net.blobs["data"].data.shape}
        )
        self.transformer.set_transpose("data", (2, 0, 1))
        self.transformer.set_mean("data", np.array([60, 60, 60]))  # mean pixel
        # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_raw_scale("data", 255)

        self.categories, self.indices, self.classes = self.get_names(
            labelmap_file
        )

    def class_vec_to_tags(self, class_vec):

        tags = []
        for category, indices, classes in zip(
            self.categories, self.indices, self.classes
        ):
            confidences = [class_vec[i] for i in indices]
            idx = np.argmax(confidences)
            tags.append(str(classes[idx]))

        return tags

    def detect(self, image, confidence):
        """
        SSD detection
        """
        # set net to batch size of 1

        orig_image_width = 2048
        orig_image_height = 512

        image_resized = cv2.resize(
            image, (self.image_resize_width, self.image_resize_height)
        )

        transformed_image = self.transformer.preprocess("data", image_resized)
        self.net.blobs["data"].data[...] = transformed_image

        # Forward pass.

        detections = self.net.forward()["detection_out"]

        num_states = 5

        # Parse the outputs.
        det_conf = detections[0, 0, :, 2]
        det_states = detections[0, 0, :, 3 : 3 + num_states + 1]
        det_xmin = detections[0, 0, :, 3 + num_states + 1]
        det_ymin = detections[0, 0, :, 3 + num_states + 2]
        det_xmax = detections[0, 0, :, 3 + num_states + 3]
        det_ymax = detections[0, 0, :, 3 + num_states + 4]
        det_states[:, 0] = 0.0

        result = []

        for i in range(len(det_xmin)):
            tags = []
            if np.max(det_conf[i]) > confidence:
                xmin = int(
                    round(det_xmin[i] * orig_image_width)
                )  # xmin = int(round(top_xmin[i] * image.shape[1]))
                ymin = int(
                    round(det_ymin[i] * orig_image_height)
                )  # ymin = int(round(top_ymin[i] * image.shape[0]))
                xmax = int(
                    round(det_xmax[i] * orig_image_width)
                )  # xmax = int(round(top_xmax[i] * image.shape[1]))
                ymax = int(
                    round(det_ymax[i] * orig_image_height)
                )  # ymax = int(round(top_ymax[i] * image.shape[0]))
                score = np.sum(det_conf[i])
                tags.extend(self.class_vec_to_tags(det_states[i]))
                result.append([xmin, ymin, xmax, ymax, tags, score])
        return result

    def get_names(self, name_file):
        categories = []
        indices = []
        names = []
        with open(name_file) as json_data:
            d = json.load(json_data)
            try:
                if len(d["categories"][0]["indices"]) != len(
                    d["categories"][0]["names"]
                ):
                    raise (
                        AssertionError(
                            "ERROR: Indices and Names should have the same length!"
                        )
                    )
                for category in d["categories"]:
                    categories.append(category["category"].encode("UTF8"))
                    indices.append(category["indices"])
                    names.append([x.encode("UTF8") for x in category["names"]])
                return categories, indices, names
            except ValueError:
                print("ERROR: File Format seems not be correct!")


def main(args):

    # Load Caffe Net
    detection = CaffeDetection(
        args.gpu_id,
        args.deploy_file,
        args.caffemodel_file,
        args.image_resize_width,
        args.image_resize_height,
        args.predictionmap_file,
    )

    # open test file in yml format
    database = driveu_dataset.DriveuDatabase(args.test_file)
    database.open("")
    cnt = 0
    # evaluate all images

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    widgets = [progressbar.Percentage(), progressbar.Bar()]
    print("\nGoing through all images")
    bar = progressbar.ProgressBar(
        widgets=widgets, max_value=len(database.images)
    ).start()

    for idx, img in enumerate(database.images):

        cnt += 1

        # get 8 bit color image from database
        status, img_color_orig = img.getImage()

        img_color = img_color_orig[0:512, 0:2048]

        # save original image size
        orig_img_width = len(img_color[0])
        orig_img_height = len(img_color)

        # detect with ssd
        result = detection.detect(img_color, args.confidence)
        bar.update(idx)

        for item in result:

            x1 = item[0]
            y1 = item[1]
            x2 = item[2]
            y2 = item[3]

            # Colorize depending on the state
            if "red_yellow" in item[4]:
                cv2.rectangle(
                    img_color_orig, (x1, y1), (x2, y2), (0, 165, 255), 2
                )
            elif "red" in item[4]:
                cv2.rectangle(
                    img_color_orig, (x1, y1), (x2, y2), (0, 0, 255), 2
                )
            elif "green" in item[4]:
                cv2.rectangle(
                    img_color_orig, (x1, y1), (x2, y2), (0, 255, 0), 2
                )
            elif "yellow" in item[4]:
                cv2.rectangle(
                    img_color_orig, (x1, y1), (x2, y2), (0, 255, 255), 2
                )
            else:
                cv2.rectangle(
                    img_color_orig, (x1, y1), (x2, y2), (255, 255, 255), 2
                )

        cv2.imshow("SSD DTLD Results", img_color_orig)
        cv2.waitKey(0)
    bar.finish()


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument(
        "--predictionmap_file",
        default="/home/muejul3/git_repos/ssd/caffe/data/tla/labelmap_tla_daimler_binary_red_negatives.prototxt",
    )
    parser.add_argument("--image_resize_width", default=2048, type=int)
    parser.add_argument("--image_resize_height", default=512, type=int)
    parser.add_argument("--confidence", default=0.5, type=float)
    parser.add_argument(
        "--deploy_file",
        default="/home/muejul3/git_repos/ssd/caffe/experiments/TLA_TRAIN_BINARY_RED_NEGATIVES/conv43_conv53/models/deploy.prototxt",
    )
    parser.add_argument(
        "--caffemodel_file",
        default="/home/muejul3/git_repos/ssd/caffe/experiments/TLA_TRAIN_BINARY_RED_NEGATIVES/conv43_conv53/models/TLA_SSD_TLA_1024x512_iter_60000.caffemodel",
    )
    parser.add_argument("--test_file", default="")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

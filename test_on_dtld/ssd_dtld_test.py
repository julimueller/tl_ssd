from __future__ import print_function

__author__ = "Julian Mueller and Klaus Dietmayer"
__maintainer__ = "Julian Mueller"
__email__ = "julian.mu.mueller@daimler.com"

import argparse
import json
import logging
import sys
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np

# if import caffe fails test export PYTHONPATH=<ssd_dir>/ssd/python:$PYTHONPATH
# and make sure you complied with make pycaffe before
import caffe
import progressbar
# if import fails please clone https://github.com/julimueller/dtld_parsing and
# install via python setup.py install
from dtld_parsing.driveu_dataset import DriveuDatabase

import os
os.environ["GLOG_minloglevel"] = "2"


# Make sure that caffe is on the python path:
caffe_root = "./"
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, "python"))

# Logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s:"
           " %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# DEFINITIONS
RGB_MEAN = [60, 60, 60]
IMAGE_MAX_VALUE = 255
SSD_INPUT_IMAGE_WIDTH = 2048
SSD_INPUT_IMAGE_HEIGHT = 512


class LabelMap:
    """
    This class loads a labelmap which transforms a vector of floats into the
    winning class
    """

    def __init__(
        self, label_map_file_path: str = "prediction_map_ssd_states.json"
    ):
        """
        LabelMap initialization and loading

        Args:
            label_map_file_path(str): Labelmap file path
        """
        self.label_map_file_path = label_map_file_path

        # load it
        self.__load__()

    def __load__(self):
        """
        Loads label map.
        """
        self.categories = []
        self.indices = []
        self.names = []
        with open(self.label_map_file_path) as json_data:
            logging.info("Loading labelmap")

            d = json.load(json_data)
            try:
                if len(d["categories"][0]["indices"]) != len(
                    d["categories"][0]["names"]
                ):
                    raise (
                        AssertionError(
                            "ERROR: Indices and Names should have"
                            "the same length!"
                        )
                    )
                for category in d["categories"]:
                    self.categories.append(category["category"])
                    self.indices.append(category["indices"])
                    self.names.append([x for x in category["names"]])
            except ValueError:
                print("ERROR: File Format seems not be correct!")

        logging.info("Labelmap successfully loaded")

    def class_vec_to_tags(self, class_vec: list):
        """
        This method converts a list of float confidences into the winning
        class by using the argmax

        Args:
            class_vec(list): list of floats,
            e.g. [0.01, 0.12, 0.55, 0.02, 0.01, 0.11]

        Returns:
            winning class as tag, e.g. [0.01, 0.12, 0.55, 0.02, 0.01, 0.11]
            -> idx: 2, tag: red
        """
        tags = []
        for category, indices, names in zip(
            self.categories, self.indices, self.names
        ):
            confidences = [class_vec[i] for i in indices]
            idx = np.argmax(confidences)
            tags.append(str(names[idx]))

        return tags

    def get_number_classes(self, category: str):
        """
        This method returns the number of classes per category

        Args:
            category(str): name of category
        Returns:
            int: Number of classes of category
        """
        idx_category = self.categories.index(category)
        return len(self.names[idx_category])


class CaffeDetection:
    """
    This class loads a trained caffe network and initializes a transformer
    converting which can convert an input image.
    """

    def __init__(
        self,
        gpu_id: int = 0,
        model_def: str = "../prototxt/deploy.prototxt",
        model_weights: str = "../caffemodel/SSD_DTLD_iter_90000.caffemodel",
        image_resize_width: int = SSD_INPUT_IMAGE_WIDTH,
        image_resize_height: int = SSD_INPUT_IMAGE_HEIGHT,
        labelmap_file_path: str = "prediction_map_ssd_states.json",
    ):
        """
        Initilize CaffeDetection

        Args:
            gpu_id(int): GPU ID. If your workstation only has only one set 0
            model_def(str): prototxt file path of deploy network
            model_weights(str): caffemodel file
            image_resize_width: width of input image
            image_resize_height: height of input image
            labelmap_file_path: path of labelmap for converting state indices

        """
        # Set GPU ID and GPU mode
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()

        self.image_resize_width = image_resize_width
        self.image_resize_height = image_resize_height
        # Load the net in the test phase for inference, and configure
        # input preprocessing.
        self.net = caffe.Net(
            model_def,  # defines the structure of the model
            model_weights,  # contains the trained weights
            caffe.TEST,
        )
        # Input preprocessing: 'data' is the name of the input
        # blob == net.inputs[0]
        self.transformer = caffe.io.Transformer(
            {"data": self.net.blobs["data"].data.shape}
        )
        # Opencv is BGR
        self.transformer.set_transpose("data", (2, 0, 1))
        # Set image mean --> DTLD is (60, 60, 60)
        # if you apply tl_ssd to another dataset please determine the dataset
        #  mean and test the performance by changing RGB_MEAN
        self.transformer.set_mean("data", np.array(RGB_MEAN))
        # The reference model operates on images in [0,255] range
        # instead of [0,1]
        self.transformer.set_raw_scale("data", IMAGE_MAX_VALUE)
        # Get names and indices for state prediction
        self.label_map = LabelMap(labelmap_file_path)
        # Get number of states
        self.num_states = self.label_map.get_number_classes(category="State")

    def parse_network_output(self, output, confidence_threshold: float = 0.5):
        """
        This method takes the raw SSD network output and returns a list of
        dictionaries

        Args:
            output(np.array): Output of detection_output_layer.count()
            confidence_threshold(float): Confidence threshold. Detections with
            lower confidence are neglected

        Returns:
            List of detections as dict

        """

        # Parse the outputs.
        confidences = output[0, 0, :, 2]
        state_confidences = output[0, 0, :, 3: 3 + self.num_states]
        xmin_coordinates = output[0, 0, :, 3 + self.num_states]
        ymin_coordinates = output[0, 0, :, 3 + self.num_states + 1]
        xmax_coordinates = output[0, 0, :, 3 + self.num_states + 2]
        ymax_coordinates = output[0, 0, :, 3 + self.num_states + 3]

        result = []

        for i in range(len(xmin_coordinates)):
            tags = []
            if confidences[i] >= confidence_threshold:
                xmin = int(round(xmin_coordinates[i] * SSD_INPUT_IMAGE_WIDTH))
                ymin = int(round(ymin_coordinates[i] * SSD_INPUT_IMAGE_HEIGHT))
                xmax = int(round(xmax_coordinates[i] * SSD_INPUT_IMAGE_WIDTH))
                ymax = int(round(ymax_coordinates[i] * SSD_INPUT_IMAGE_HEIGHT))
                tags.extend(
                    self.label_map.class_vec_to_tags(state_confidences[i])
                )
                detection = {
                    "xmin": xmin,
                    "xmax": xmax,
                    "ymin": ymin,
                    "ymax": ymax,
                    "confidence": confidences[i],
                    "tags": tags,
                }
                result.append(detection)
        return result

    def detect(self, image: np.array, confidence_threshold: float = 0.5):
        """
        This method detects traffic lights in the input image for a given
        confidence threshold

        Args:
            image(np.array): Numpy array image in BGR format (OpenCV)
            confidence(float): confidence threshold. Only detections with
            confidence >= confidence_threshold are returned
        """
        # set net to batch size of 1

        image_resized = cv2.resize(
            image, (self.image_resize_width, self.image_resize_height)
        )

        transformed_image = self.transformer.preprocess("data", image_resized)
        self.net.blobs["data"].data[...] = transformed_image

        # Forward pass.

        detections = self.net.forward()["detection_out"]

        return self.parse_network_output(detections, confidence_threshold)


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

    # Open test file in yml format
    database = DriveuDatabase(args.test_file)
    database.open("")

    # Progressbar
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    widgets = [progressbar.Percentage(), progressbar.Bar()]
    print("\nGoing through all images")
    bar = progressbar.ProgressBar(
        widgets=widgets, max_value=len(database.images)
    ).start()

    # create axes
    ax1 = plt.subplot(111)

    for idx, img in enumerate(database.images):

        # Get 8 bit color image from database
        status, img_color_orig = img.get_image()
        # Crop image
        img_color = img_color_orig[0:512, 0:2048]
        # Detect with ssd
        result = detection.detect(img_color, args.confidence)
        # Plot each detection
        for item in result:

            xmin = item["xmin"]
            ymin = item["ymin"]
            xmax = item["xmax"]
            ymax = item["ymax"]

            # Colorize depending on the state
            if "red_yellow" in item["tags"]:
                cv2.rectangle(
                    img_color_orig,
                    (xmin, ymin),
                    (xmax, ymax),
                    (0, 165, 255),
                    2,
                )
            elif "red" in item["tags"]:
                cv2.rectangle(
                    img_color_orig, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2
                )
            elif "green" in item["tags"]:
                cv2.rectangle(
                    img_color_orig, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2
                )
            elif "yellow" in item["tags"]:
                cv2.rectangle(
                    img_color_orig,
                    (xmin, ymin),
                    (xmax, ymax),
                    (0, 255, 255),
                    2,
                )
            else:
                cv2.rectangle(
                    img_color_orig,
                    (xmin, ymin),
                    (xmax, ymax),
                    (255, 255, 255),
                    2,
                )
        bar.update(idx)
        # Because of the weird qt error in gui methods in opencv-python >= 3
        # imshow does not work in some cases. You can try it by yourself.
        # cv2.imshow("SSD DTLD Results", img_color_orig)
        # cv2.waitKey(0)
        img_rgb = img_color_orig[..., ::-1]
        if idx == 0:
            im1 = ax1.imshow(img_rgb)
        plt.ion()
        im1.set_data(img_rgb)
        plt.pause(0.1)
        plt.draw()
    bar.finish()


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument(
        "--predictionmap_file",
        default="prediction_map_ssd_states.json",
    )
    parser.add_argument("--image_resize_width", default=2048, type=int)
    parser.add_argument("--image_resize_height", default=512, type=int)
    parser.add_argument("--confidence", default=0.5, type=float)
    parser.add_argument(
        "--deploy_file",
        default="../prototxt/deploy.prototxt",
    )
    parser.add_argument(
        "--caffemodel_file",
        default="../caffemodel/SSD_DTLD_iter_90000.caffemodel",
    )
    parser.add_argument("--test_file", default="")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

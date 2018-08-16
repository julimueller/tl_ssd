# IN PROGRESS: TL-SSD: Detecting Traffic Lights by Single Shot Detection
This respository provides code for our single shot detection method to detect traffic lights. Paper is accepted for ITSC2018. The code builds upon the official SSD-Caffe-Repository (see here https://github.com/weiliu89/caffe).

Following steps are recommended:

1. Download and build the original caffe-ssd repository according to their instructions.

2. Replace the original files by our modifications and build again.

The code can be used in two ways:

1. Use TL-SSD only to detect smaller objects with our adaptions in the prior box layer. This way you have to set the following parameters:

2. Use TL-SSD to detect smaller objects AND use the additional state prediction. This way you have to set the following parameters:

How to use modified prior box layer in caffe prototxt format:

```
layer {
  name: "inception_b4_concat_norm_mbox_priorbox"
  type: "PriorBox"
  bottom: "inception_b4_concat_norm"
  bottom: "data"
  top: "inception_b4_concat_norm_mbox_priorbox"
  prior_box_param {
    min_size: 7
    min_size: 10
    min_size: 15
    min_size: 25
    min_size: 35
    min_size: 50
    min_size: 70
    aspect_ratio: 0.3
    flip: false
    clip: false
    variance: 0.1
    variance: 0.1
    variance: 0.2
    variance: 0.2
    offset_w: 0.2
    offset_w: 0.4
    offset_w: 0.6
    offset_w: 0.8
    offset_h: 0.5
  }
}
```

Offsets are "shiftings" of the prior box in the feature cell. offset_w in x direction, offset_h in y direction. Take a look into the code for detailed understanding. We modified further small things, which were hard-coded in the original implementation, such as one default prior box with aspect ratio of 1. Furthermore, we do not use the max_size parameters. 
In progress.

Stay tuned

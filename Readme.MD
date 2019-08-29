# TL-SSD: Detecting Traffic Lights by Single Shot Detection
This respository provides code for our single shot detection method to detect traffic lights. Paper is accepted for ITSC2018. The code builds upon the official SSD-Caffe-Repository (see here https://github.com/weiliu89/caffe). It presents a prior box/anchor box adaptation to increase recall on small objects using SSD.

[![IMAGE ALT TEXT HERE](https://github.com/julimueller/tl_ssd/blob/master/vid/vid_thumb.png)](https://www.youtube.com/watch?v=ei0FkLMJs04&feature=youtu.be)

Following steps are recommended:

1. Download and build the original caffe-ssd repository according to their instructions.

2. Replace the original files by our modifications and build again.

The code can be used in two ways:

1. Use TL-SSD only to detect smaller objects with our adaptions in the prior box layer WITHOUT a subsequent state detection. 

2. Use TL-SSD to detect smaller objects AND use the additional state prediction. 

## Only use the prior box adaptions
A possible usage of the MultiBoxLayer looks as follows:
```
layer {
  name: "mbox_loss"
  type: "MultiBoxLoss"
  bottom: "mbox_loc"
  bottom: "mbox_conf"
  bottom: "mbox_priorbox"
  bottom: "label"
  top: "mbox_loss"
  include {
    phase: TRAIN
  }
  propagate_down: true
  propagate_down: true
  propagate_down: false
  propagate_down: false
  propagate_down: true
  loss_param {
    normalization: VALID
  }
  multibox_loss_param {
    loc_loss_type: SMOOTH_L1
    conf_loss_type: SOFTMAX
    loc_weight: 1.0
    num_classes: 2
    share_location: true
    match_type: PER_PREDICTION
    overlap_threshold: 0.3
    use_prior_for_matching: true
    background_label_id: 0
    use_difficult_gt: true
    neg_pos_ratio: 3.0
    neg_overlap: 0.5
    code_type: CENTER_SIZE
    ignore_cross_boundary_bbox: false
    mining_type: MAX_NEGATIVE
    do_state_prediction: false
  }
}

```

Prior Box Adaptions in prototxt:

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
## Use the prior box adaptions AND state detection for TLD

```
layer {
  name: "mbox_loss"
  type: "MultiBoxLoss"
  bottom: "mbox_loc"
  bottom: "mbox_conf"
  bottom: "mbox_priorbox"
  bottom: "label"
  bottom: "mbox_state"
  top: "mbox_loss"
  include {
    phase: TRAIN
  }
  propagate_down: true
  propagate_down: true
  propagate_down: false
  propagate_down: false
  propagate_down: true
  loss_param {
    normalization: VALID
  }
  multibox_loss_param {
    loc_loss_type: SMOOTH_L1
    conf_loss_type: SOFTMAX
    loc_weight: 1.0
    num_classes: 2
    share_location: true
    match_type: PER_PREDICTION
    overlap_threshold: 0.3
    use_prior_for_matching: true
    background_label_id: 0
    use_difficult_gt: true
    neg_pos_ratio: 3.0
    neg_overlap: 0.5
    code_type: CENTER_SIZE
    ignore_cross_boundary_bbox: false
    mining_type: MAX_NEGATIVE
    state_weight: 1.0
    do_state_prediction: true
    num_states: 6
    background_state_id: 0
    state_digit: 4
    state_loss_type: LOGISTIC
  }
}

```

Additional parameters compared to the original SSD are

1. Set boolean for state prediction as true

```
do_state_prediction: true

```
2. Specify the state weight b. Overall loss is calculated as L = L_conf + a * L_loc + b * L_state

```
state_weight: 1.0

```
3. Specify the number of states. Please note that an additional background state is predicted as well. In other words, if your dataset contains the states red, yellow, green, you have to set the num_states to 3 + 1 = 4.

```
num_states: 6

```
4. Specify the background state id. See explanations of 3. 
```
background_state_id: 0

```
5. Specify the state digit. 
```
state_digit: 4
```
This refers to the way the labels are prepared. Original SSD used lmdb format created from lists of .jpg and .txt. The label files could look like the following when u use the DriveU Traffic Light Dataset (DTLD), which contain class labels consisting of 6 digits, whereas the 5th digit contains state information

```
class  xmin xmax ymin ymax
112340 1073 234 1081  257
132340 1251 298 1256  318
112340 795  323 799   339
122310 1127 347 1129  350
```
The space digit parameter specifies, at which position of the class-id the state is encoded. Of course we could also have changed the complete data loading section the enhance the label format by an additional state value but this had caused way more implementation effort and changes in the SSD data structures. If you want to use another dataset only containing color information, you could do it like that

How to use modified prior box layer in caffe prototxt format:
```
class  xmin xmax ymin ymax
1 1073 234 1081  257
2 1251 298 1256  318
3 795  323 799   339
```
where 1 = red, 2 = yellow and 3 = green with state_digit: 0.
6. Specify the state loss type. The available loss functions are equal to the confidence loss functions. However, we recommend the logistic loss.
```
state_loss_type: LOGISTIC

```
## Test our provided caffemodel on the DriveU Traffic Light Dataset
1. Download the dataset http://traffic-light-data.de/
```
cd test_on_dtld

python ssd_dtld_test.py --predictionmap_file <tl_ssd_path>/test_on_dtld/prediction_map_ssd_states.json --confidence 0.2 --deploy_file <tl_sdd_path>/prototxt/deploy.prototxt --caffemodel_file <tl_sdd_path>caffemodel/SSD_DTLD_iter_90000.caffemodel --test_file <db_dir_path>/DTLD_test.yml
```
Do not forget to update the absolute caffe ssd path in ssd_dtld_test.py before importing

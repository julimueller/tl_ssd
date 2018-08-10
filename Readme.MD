# Detecting Traffic Lights by Single Shot Detection
This respository provides code for our single shot detection method to detect traffic lights. Paper is accepted for ITSC2018. The code builds upon the official SSD-Caffe-Repository (see here https://github.com/weiliu89/caffe).

Following steps are recommended:

1. Download and build the original caffe-ssd repository according to their instructions.

2. Replace the original files by our modifications and build again.

The code can be used in two ways:

1. Use TL-SSD only to detect smaller objects with our adaptions in the prior box layer. This way you have to set the following parameters:

2. Use TL-SSD to detect smaller objects AND use the additional state prediction. This way you have to set the following parameters:


In progress. Stay tuned

# Unet_segmentaion
Segmentation for medical with U-net model
* Using tensorflow 2.0 and later 
* U-net model with backbone resnet (https://github.com/qubvel/segmentation_models)
# Usage 
* ./config store labelme data 
* For custom classes, change Dataset object
* Using data_preprocessing.py to process labelme json file
# Changelog 
* 2021/01/26: Change mask to one hot econding
* 2021/02/02: Add TF_record and data augmentation and callback function

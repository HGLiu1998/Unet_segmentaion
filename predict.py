import gc

from tensorflow.python.ops.gen_math_ops import IsInf
from segmentation_models import Unet
from PIL import Image
from keras.optimizers import *
import numpy as np
import os, sys, time, random, math
import skimage.io
import skimage.transform as trans
import re
import cv2
import matplotlib
import matplotlib.pyplot as plt
import segmentation_models as sm
import yaml
import keras
from keras.models import save_model
from keras.preprocessing.image import ImageDataGenerator
import glob
import albumentations as A
import tensorflow as tf
import colorsys
from tensorflow.keras.models import load_model
import datetime
# from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
import multiprocessing

sm.set_framework('tf.keras')
BACKBONE = 'resnet50'
BATCH_SIZE = 6
CLASSES = ['Burn']
LR = 0.0001
EPOCHS = 400
#label = {"_background_": 0, "Ulceration": 1, "slough tissue": 2, "Re-ep": 3, "Granulation": 4, "Eschar": 5}

class Dataset():
    CLASSES_Ul = ['background, Ulceration', 'Re-ep']
    CLASSES_Other = ['background', 'slough-tissue', 'Granulation', 'Eschar']
    def __init__(self, config_path = './config/', only_Ul=True, augmentation=None, preprocessing=None):
        self.config_path = config_path
        if only_Ul:
            self.classes = 3
            self.color2index = {
                (0, 0, 0) : 0, # _background_
                (128, 0, 0) : 1, # Ul
                (0, 128, 0) : 1, # slogh 
                (0, 0, 128) : 2, # Re-ep
                (128,128,0) : 1, # Granulation
                (128,0,128) : 1 # Eschar
            }
        else:
            self.classes = 4
            self.color2index = {
                (0, 0, 0) : 0, # _background_
                (128, 0, 0) : 0, # Ul
                (0, 128, 0) : 1, # slogh 
                (0, 0, 128) : 0, # Re-ep
                (128,128,0) : 2, # Granulation
                (128,0,128) : 3 # Eschar
            }

    def rgb2mask(self, img, classes):
        assert len(img.shape) == 3
        height, width, ch = img.shape
        assert ch == 3
        W = np.power(256, [[0],[1],[2]])
        img_id = img.dot(W).squeeze(-1) 
        values = np.unique(img_id)
        mask = np.zeros((height,width,classes))
        for i, c in enumerate(values):
            one_hot = [0. for j in range(classes)]
            one_hot[self.color2index[tuple(img[img_id==c][0])]] = 1.0
            mask[img_id==c] = one_hot
        return mask

    def load_data(self, data=None):
        raw_dataset = tf.data.TFRecordDataset(
            filenames='train.tfrecords')
        # Parse the input tf.train.Example proto using the dictionary above.
        image_feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'mask': tf.io.FixedLenFeature([], tf.string),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64)
        }

        def _parse_image_function(example_proto):
            # Parse the input tf.train.Example proto using the dictionary above.
            return tf.io.parse_single_example(example_proto, image_feature_description)

        n_map_threads = multiprocessing.cpu_count()
        parsed_dataset = raw_dataset.map(
            _parse_image_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.total_data = []
        self.total_mask = []
        self.colored_masks = []
        for parsed_record in parsed_dataset:
            height = int(parsed_record['height'])
            width = int(parsed_record['width'])
            image_str = parsed_record['image'].numpy()
            mask_str = parsed_record['mask'].numpy()
            image = np.frombuffer(image_str, dtype=np.uint8)
            image = image.reshape((height, width, 3))
            mask = np.frombuffer(mask_str, dtype=np.uint8)
            mask = mask.reshape((height, width, 3))
            self.colored_masks.append(mask)
            # cv2.imwrite('output_mask.png', mask)
            mask = self.rgb2mask(mask, self.classes)
            # cv2.imwrite('output_img.jpg', image)
            self.total_data.append(image)
            self.total_mask.append(mask)
    
    def preprocessing(self, image, label):
        image = tf.cast(image, tf.float64) / 255.0
        return image, label

    def spilt_set(self, split):
        
        training_data = np.squeeze(np.array(self.total_data))
        training_mask = np.squeeze(np.array(self.total_mask))
        

        data_size = training_data.shape[0]
        split_size = int(data_size * split)
        self.split_size = split_size

        # data split
        training_dataset = tf.data.Dataset.from_tensor_slices(
            (training_data[:-split_size], training_mask[:-split_size]))
        validation_dataset = tf.data.Dataset.from_tensor_slices(
            (training_data[-split_size:], training_mask[-split_size:]))
        # data preprocessing and augmentation
        training_dataset = training_dataset.map(
            self.preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
        training_dataset = training_dataset.shuffle(100)
        training_dataset = training_dataset.batch(BATCH_SIZE)
        training_dataset = training_dataset.prefetch(
            tf.data.experimental.AUTOTUNE
            )
        validation_dataset = validation_dataset.map(
            self.preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
        validation_dataset = validation_dataset.batch(BATCH_SIZE)
        validation_dataset = validation_dataset.prefetch(
            tf.data.experimental.AUTOTUNE
            )
        return training_dataset, validation_dataset

def mask2rgb(mask):
    mask = np.argmax(mask, axis=-1)
    #print(mask.shape)
    output = np.zeros((512,512,3), dtype=np.uint8)
    for i in range(512):
        for j in range(512):
            if mask[i,j] == 0:
                output[i,j] = [0,0,0]
            elif mask[i,j] == 1:
                output[i,j] = [128,0,0]
            else :
                output[i,j] = [0,0,128]
    return output
def main():
    dataset = Dataset('./config/')

    metrics = [
        sm.metrics.IOUScore(threshold=0.5), 
        sm.metrics.FScore(threshold=0.5), 
        sm.metrics.Recall(threshold=0.5), 
        sm.metrics.Precision(threshold=0.5),
        sm.metrics.Accuracy(threshold=0.5),
    ]

    dataset.load_data()
    split_ratio = 0.1
    training_dataset, validation_dataset = dataset.spilt_set(split_ratio)     

    unet_model = tf.keras.models.load_model("Unet_Ulceration_20210218-223943.h5", custom_objects={
        'dice_loss':sm.losses.DiceLoss(), 
        'iou_score':sm.metrics.IOUScore(threshold=0.5),
        'f1-score':sm.metrics.FScore(threshold=0.5),
        'recall':sm.metrics.Recall(threshold=0.5), 
        'precision':sm.metrics.Precision(threshold=0.5),
        'accuracy':sm.metrics.Accuracy(threshold=0.5)
        })
        
    #result = unet_model.evaluate(validation_dataset)
    #print(dict(zip(unet_model.metrics_names, result)))

    for batch_idx, element in enumerate(validation_dataset):
        idx = batch_idx * BATCH_SIZE
        image_batch = element[0].numpy()
        mask_batch = element[1].numpy()
        size = image_batch.shape[0]
        os.mkdir('val/val{}'.format(idx))
        for i in range(size):
            image = (image_batch[i] * 255.0).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite('val/val{}/{}_Input.png'.format(idx, i), image)
            mask = mask_batch[i]
            mask = mask2rgb(mask)
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
            cv2.imwrite('val/val{}/{}_Label.png'.format(idx,i), mask)
            #print(element[0])
        prediction = unet_model.predict(element[0])
        prediction = np.argmax(prediction, axis=-1)
        print(prediction.shape)
        for n in range(size):
            output = np.zeros((512,512,3), dtype=np.uint8)
            for i in range(512):
                for j in range(512):
                    if prediction[n,i,j] == 0:
                        output[i,j] = [0,0,0]
                    elif prediction[n,i,j] == 1:
                        output[i,j] = [128,0,0]
                    else :
                        output[i,j] = [0,0,128]
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            cv2.imwrite('val/val{}/{}_Output.png'.format(idx, n), output)            

    # # np.savetxt('prediction.txt', prediction)
    return unet_model

if __name__ == '__main__':
    main()


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
BATCH_SIZE = 5

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

    def load_data(self, data=None):
        self.testing_data = []
        if data != None:
            raw_dataset = tf.data.TFRecordDataset(
                filenames=data)
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
            for parsed_record in parsed_dataset:
                height = int(parsed_record['height'])
                width = int(parsed_record['width'])
                image_str = parsed_record['image'].numpy()
                mask_str = parsed_record['mask'].numpy()
                image = np.frombuffer(image_str, dtype=np.uint8)
                image = image.reshape((height, width, 3))
                mask = np.frombuffer(mask_str, dtype=np.uint8)
                mask = mask.reshape((height, width, 3))
        
                self.testing_data.append(image)
        else:
            dirs = os.listdir(self.config_path)
            for d in dirs:
                print(d)
                image = cv2.imread(self.config_path+d)
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w, c = image.shape
                image = tf.image.resize(image, (512,512), method='nearest').numpy()

                self.testing_data.append(image)   

        testing_data = np.squeeze(np.array(self.testing_data))
        testing_dataset = tf.data.Dataset.from_tensor_slices((testing_data))
        testing_dataset = testing_dataset.map(self.preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        testing_dataset = testing_dataset.batch(len(self.testing_data))
        testing_dataset = testing_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return testing_dataset
    
    def preprocessing(self, image):
        image = tf.cast(image, tf.float64) / 255.0
        return image

def mask2rgb(mask, only_Ul = False):
    mask = np.argmax(mask, axis=-1)
    #print(mask.shape)
    output = np.zeros((512,512,3), dtype=np.uint8)
    for i in range(512):
        for j in range(512):
            if only_Ul:
                if mask[i,j] == 0:
                    output[i,j] = [0,0,0]
                elif mask[i,j] == 1:
                    output[i,j] = [128,0,0]
                else :
                    output[i,j] = [0,0,128]
            else:
                if mask[i,j] == 0:
                    output[i,j] = [0,0,0]
                elif mask[i,j] == 1:
                    output[i,j] = [0,128,0]
                elif mask[i, j] == 2:
                    output[i,j] = [128,128,0]
                else :
                    output[i,j] = [128,0,128]
    return output

def area_count(percentage, x=38):
    return percentage * 6400 * pow(x/76, 2)

def main():
    dataset = Dataset(config_path='./testing/')

    metrics = [
        sm.metrics.IOUScore(threshold=0.5), 
        sm.metrics.FScore(threshold=0.5), 
        sm.metrics.Recall(threshold=0.5), 
        sm.metrics.Precision(threshold=0.5),
        sm.metrics.Accuracy(threshold=0.5),
    ]

    testing_dataset = dataset.load_data()
    

    slough_model = tf.keras.models.load_model("20210222-013609 Unet_Slough.h5", custom_objects={
        'dice_loss':sm.losses.DiceLoss(), 
        'iou_score':sm.metrics.IOUScore(threshold=0.5),
        'f1-score':sm.metrics.FScore(threshold=0.5),
        'recall':sm.metrics.Recall(threshold=0.5), 
        'precision':sm.metrics.Precision(threshold=0.5),
        'accuracy':sm.metrics.Accuracy(threshold=0.5)
        })
    
    ul_model = tf.keras.models.load_model("Unet_Ulceration-20210223-013304.h5", custom_objects={
        'dice_loss':sm.losses.DiceLoss(), 
        'iou_score':sm.metrics.IOUScore(threshold=0.5),
        'f1-score':sm.metrics.FScore(threshold=0.5),
        'recall':sm.metrics.Recall(threshold=0.5), 
        'precision':sm.metrics.Precision(threshold=0.5),
        'accuracy':sm.metrics.Accuracy(threshold=0.5)
        })
    
    for batch_idx, element in enumerate(testing_dataset):
        image_batch = element.numpy()
        size = image_batch.shape[0]
        os.mkdir('test/val{}'.format(batch_idx))
        for i in range(size):
            image = (image_batch[i] * 255.0).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite('test/val{}/{}_Input.png'.format(batch_idx, i), image)
                ## Slough model predict
        slough_prediction = slough_model.predict(element)
        slough_prediction = np.argmax(slough_prediction, axis=-1)
        ul_prediction = ul_model.predict(element)
        ul_prediction = np.argmax(ul_prediction, axis=-1)
        for n in range(size):
            Area = [0. for i in range(6)]
            slough_output = np.zeros((512,512,3), dtype=np.uint8)
            ul_output = np.zeros((512,512,3), dtype=np.uint8)
            for i in range(512):
                for j in range(512):
                    if slough_prediction[n,i,j] == 0:
                        slough_output[i,j] = [0,0,0]
                    elif slough_prediction[n,i,j] == 1:
                        slough_output[i,j] = [128,128,0]
                        Area[2] += 1
                    elif slough_prediction[n,i,j] == 2:
                        slough_output[i,j] = [128,0,0]
                        Area[4] += 1
                    elif slough_prediction[n,i,j] == 3:
                        slough_output[i,j] = [0,0,128]
                        Area[5] += 1

            for i in range(512):
                for j in range(512):
                    if ul_prediction[n,i,j] == 0:
                        ul_output[i,j] = [0,0,0]
                    elif ul_prediction[n,i,j] == 1:
                        Area[1] += 1
                        ul_output[i,j] = [0,128,0]
                    elif ul_prediction[n,i,j] == 2:
                        Area[3] += 1
                        ul_output[i,j] = [128,0,128]
            slough_output = cv2.cvtColor(slough_output, cv2.COLOR_RGB2BGR)
            cv2.imwrite('test/val{}/{}_slough.png'.format(batch_idx, n), slough_output)   
            ul_output = cv2.cvtColor(ul_output, cv2.COLOR_RGB2BGR)
            cv2.imwrite('test/val{}/{}_Ul.png'.format(batch_idx, n), ul_output)
            print(Area[2]/Area[1])
            area = area_count(Area[2]/(512*512))
            print("Area{}: {}".format(n, area))

    # # np.savetxt('prediction.txt', prediction)

if __name__ == '__main__':
    main()
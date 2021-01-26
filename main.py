import gc
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


sm.set_framework('tf.keras')
BACKBONE = 'resnet50'
BATCH_SIZE = 8
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

    def load_data(self, split, data=None):
        
        training_data = []
        training_mask = []
        dirs = os.listdir(self.config_path)
        for d in dirs:
            image = cv2.imread(self.config_path+d+'/img.png')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, c = image.shape
            image = image[h//2-256:h//2+256, w//2-256:w//2+256]
            training_data.append(image)
            label = cv2.imread(self.config_path+d+'/label.png')
            label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
            label = label[h//2-256:h//2+256, w//2-256:w//2+256]

            label = self.rgb2mask(label, self.classes)     
            training_mask.append(label)
            
        
        training_data = np.squeeze(np.array(training_data))
        training_mask = np.squeeze(np.array(training_mask))
        
        print(training_data.shape)
        print(training_mask.shape)
        data_size = training_data.shape[0]
        split_size = int(data_size * split)
        print(split_size)
        # data split
        training_dataset = tf.data.Dataset.from_tensor_slices((training_data[:-split_size], training_mask[:-split_size]))
        validation_dataset = tf.data.Dataset.from_tensor_slices((training_data[-split_size:], training_mask[-split_size:]))
        # data preprocessing and augmentation
        training_dataset = training_dataset.map(self.preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        training_dataset = training_dataset.shuffle(100)
        training_dataset = training_dataset.batch(6)
        training_dataset = training_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        validation_dataset = validation_dataset.map(self.preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        validation_dataset = validation_dataset.shuffle(100)
        validation_dataset = validation_dataset.batch(6)
        validation_dataset = validation_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return training_dataset, validation_dataset
    
    def preprocessing(self, image, label):
        image = tf.cast(image, tf.float64) / 255.0
        return image, label
        


    

def main():
    dataset = Dataset('./config/')
    preprocess_input = sm.get_preprocessing(BACKBONE)

    model = sm.Unet(BACKBONE, classes=dataset.classes, activation='softmax', encoder_weights='imagenet')

    #model.load_weights(MODEL_PATH)
    # define optomizer
    optim = keras.optimizers.Adam(LR)

    # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss #+ (1 * focal_loss)

    # actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
    # total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

    metrics = [
        sm.metrics.IOUScore(threshold=0.5), 
        sm.metrics.FScore(threshold=0.5), 
        sm.metrics.Recall(threshold=0.5), 
        sm.metrics.Precision(threshold=0.5),
        sm.metrics.Accuracy(threshold=0.5),
    ]

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss=total_loss, metrics=metrics)
    train_dataset, validation_dataset = dataset.load_data(0.1)
    model.fit(train_dataset,batch_size=6,epochs=1000,validation_data=validation_dataset, callbacks=[tensorboard_callback])
    model.save('Unet_Ulceration.h5')

if __name__ == '__main__':
    main()


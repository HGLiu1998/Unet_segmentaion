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
import tensorflowjs as tfjs


sm.set_framework('tf.keras')
BACKBONE = 'resnet101'
BATCH_SIZE = 4
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
            self.config_path = './ul_config/'
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
            self.config_path = './other_config'
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

    def load_data(self, validation_split=0.2, testing_split=0.1, data=None):

        if data != None:
            raw_dataset = tf.data.TFRecordDataset(filenames=data)
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
                _parse_image_function, num_parallel_calls=n_map_threads)
            training_data = []
            training_mask = []

            for parsed_record in parsed_dataset:
                height = int(parsed_record['height'])
                width = int(parsed_record['width'])
                image_str = parsed_record['image'].numpy()
                mask_str = parsed_record['mask'].numpy()
                image = np.frombuffer(image_str, dtype=np.uint8)
                image = image.reshape((height, width, 3))
                mask = np.frombuffer(mask_str, dtype=np.uint8)
                mask = mask.reshape((height, width, 3))
                mask = self.rgb2mask(mask, self.classes)
                training_data.append(image)
                training_mask.append(mask)

        else:
            training_data = []
            training_mask = []
            dirs = os.listdir(self.config_path)
            for d in dirs:
                #print(d)
                image = cv2.imread(self.config_path+d+'/img.png')
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w, c = image.shape
                label = cv2.imread(self.config_path+d+'/label.png')
                label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

                #image, label = self.cropping(image, label)

                image = tf.image.resize(image, (512,512), method='nearest').numpy()
                label = tf.image.resize(label, (512,512), method='nearest').numpy()
                
                #cv2.imwrite('image.png',image)
                #cv2.imwrite('label.png',label)
                label = self.rgb2mask(label, self.classes)  

                training_data.append(image)   
                training_mask.append(label)
            
        
        training_data = np.squeeze(np.array(training_data))
        training_mask = np.squeeze(np.array(training_mask))
        data_size = training_data.shape[0]
        training_size = data_size - int(data_size * (validation_split+testing_split))
        validation_size = training_size + int(data_size * validation_split)
        print(training_size)
        print(validation_size)
        # data split
        training_dataset = tf.data.Dataset.from_tensor_slices((training_data[:training_size], training_mask[:training_size]))
        validation_dataset = tf.data.Dataset.from_tensor_slices((training_data[training_size:validation_size], training_mask[training_size:validation_size]))
        testing_dataset = tf.data.Dataset.from_tensor_slices((training_data[validation_size:], training_mask[validation_size:]))
        # data preprocessing and augmentation
        training_dataset = training_dataset.map(self.preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        #training_dataset = training_dataset.map(self.augmentation, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        training_dataset = training_dataset.shuffle(100)
        training_dataset = training_dataset.batch(5)
        training_dataset = training_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        
        validation_dataset = validation_dataset.map(self.preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        validation_dataset = validation_dataset.batch(5)
        validation_dataset = validation_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        
        testing_dataset = testing_dataset.map(self.preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        testing_dataset = testing_dataset.batch(5)
        testing_dataset = testing_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        print(testing_dataset, validation_dataset, testing_dataset)
        return training_dataset, validation_dataset, testing_dataset

    def preprocessing(self, image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    def augmentation(self, image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, 0.2)
        return image, label

def main():
   
    dataset = Dataset('./config/', True)
    preprocess_input = sm.get_preprocessing(BACKBONE)

    model = sm.Unet(BACKBONE, classes=dataset.classes, activation='softmax', encoder_weights='imagenet')

   
    optim = tf.keras.optimizers.Adam(learning_rate=0.1)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    callback = [
        tf.keras.callbacks.ReduceLROnPlateau('val_f1-score', 0.5, 5, min_lr=5e-6),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        tf.keras.callbacks.ModelCheckpoint('{}.h5'.format(log_dir), monitor='val_f1-score', save_best_only=True,save_freq=10,mode='max')
    ]
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

    #log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    
    model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss=total_loss, metrics=metrics)
    train_dataset, validation_dataset, testing_dataset = dataset.load_data(0.1)
    model.fit(train_dataset,batch_size=5,epochs=1000,validation_data=validation_dataset, callbacks=callback)
    model.evaluate(testing_dataset,batch_size=5,callbacks=callback)
    model.save('Unet_Ulceration.h5')
    tfjs.converters.save_keras_model(model, './js')

if __name__ == '__main__':
    main()


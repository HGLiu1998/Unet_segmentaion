import tensorflow as tf
import numpy as np
import cv2
import os
import base64

from tensorflow.python.ops.gen_logging_ops import image_summary

def main():

    data_dir = './ul_config/'
    tf_record = './train.tfrecords'
    dirs = os.listdir(data_dir)
    dirs.sort()

    image_filenames = [data_dir+d+'/img.png' for d in dirs]
    mask_filenames = [data_dir+d+'/label.png' for d in dirs]

    with tf.io.TFRecordWriter(tf_record) as writer:
        for image, mask in zip(image_filenames, mask_filenames):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = tf.image.resize(image, (512,512), method='nearest').numpy()
            _h, _w, c = image.shape
            image_str = image.tostring()

            mask = cv2.imread(mask)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask = tf.image.resize(mask, (512,512), method='nearest').numpy()
            mask_str = mask.tostring()

            feature = {
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_str])),
                'mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[mask_str])),
                'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[_h])),
                'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[_w]))
            }
            example = tf.train.Example(
                features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            
        writer.close()

if __name__ == '__main__':
    main()
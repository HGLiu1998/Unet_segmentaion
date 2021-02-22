import os
import fnmatch
import cv2
import numpy as np

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def main():
    data_path = './dataset3/' #superpixel dir name
    store_path = './dataset3_clean/'
    masks = find('*mask*.png', data_path)
    i = 0
    for f in masks:
        print('image index {}'.format(i))
        former = os.path.splitext(os.path.basename(f))[0]
        base_num = former.split('_')[0]
        if len(former.split('(')) != 1:
            special_str = former.split('(')[1].split(')')[0]
            image_name = data_path + \
                '{} ({}).png'.format(base_num, special_str)
            # print(former)
            print(image_name)
            image = cv2.imread(image_name)
            mask = cv2.imread(f)
            # print(image)
            cv2.imwrite(store_path + '{}.png'.format(i),image)
            cv2.imwrite(store_path + '{}_mask.png'.format(i),mask)
            # if image is None:
                # print(image_name + 'is None')

        else:
            image_name = data_path + base_num +'.png'
            print(image_name)
            image = cv2.imread(image_name)
            mask = cv2.imread(f)
            # print(image)
            cv2.imwrite(store_path + '{}.png'.format(i),image)
            cv2.imwrite(store_path + '{}_mask.png'.format(i),mask)
            # if image is None:
                # print(image_name + 'is None')
        i += 1

def color_mapping():
    data_path = './dataset3_clean/'
    masks = find('*_mask.png', data_path)
    for f in masks:
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        # color_table = {
        #     'background' : (0, 0, 0), #rgb
        #     'ulceration' : (128, 0, 0),
        #     'slough' : (0, 128, 0),
        #     'Re-ep' : (0, 0, 128),
        #     'Granulation' : (128,128,0),
        #     'Eschar' : (128,0,128)
        # }
        #slough tissue
        mask = (img == [0,255,255]).all(axis=2) 
        img[mask] = [0,128,0] #bgr
        #eschar
        mask = (img == [255,0,0]).all(axis=2) 
        img[mask] = [128,0,128]
        #granulation
        mask = (img == [0,0,255]).all(axis=2) 
        img[mask] = [0,128,128] 
        #Re-ep
        mask = (img == [255,255,255]).all(axis=2) 
        img[mask] = [128,0,0]
        cv2.imwrite(f, img)

if __name__ == '__main__':
    main()
    color_mapping()

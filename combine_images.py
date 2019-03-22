import cv2
import os
import numpy as np

input_path = 'train_img/train/input/'
label_path = 'train_img/train/label/'
combine_path = 'train_img/train/combine/'

def get_image_names():
    image_names = os.listdir(input_path)
    return image_names

def combine_images(image_names):
    for i in range(len(image_names)):
        img1 = cv2.imread(input_path + image_names[i])
        img2 = cv2.imread(label_path + image_names[i])
        combine_img = np.concatenate((img1, img2), axis=1)
        cv2.imwrite(combine_path+image_names[i], combine_img)

if __name__=="__main__":
    image_names = get_image_names()
    combine_images(image_names)

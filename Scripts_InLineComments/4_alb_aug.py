"""
For Transforms of the images and augmentation of the images
can also use the - https://github.com/albumentations-team/albumentations_examples
"""
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import albumentations as alb
#from albumentations import *
#Show image after - albumentations-
print("-Testing-------albumentations--AAA")
img = Image.open("../img_inputs/dog.jpg").convert('RGB')
img_t = preprocess(img)
#plt.imshow(np.transpose(img_t,(1, 2, 0))) 
#
transform = alb.HorizontalFlip(p=0.5)
#random.seed(7)
alb_aug_img = transform(image=img_t)['image']
print(type(alb_aug_img))
#plt.imshow(augmented_image)
#
print("-Testing-------albumentations--BBB")
#Without_PreProcess == TypeError: Invalid shape (1176, 3, 595) for image data

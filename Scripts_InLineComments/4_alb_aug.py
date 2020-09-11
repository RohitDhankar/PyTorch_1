"""
For Transforms of the images and augmentation of the images
can also use the - https://github.com/albumentations-team/albumentations_examples
"""
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import albumentations as alb
import random , cv2 

def vis_img(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)
    plt.show()

image = cv2.imread('../img_inputs/dog.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#print(type(image))#<class 'numpy.ndarray'>
#vis_img(image)    
transform = alb.HorizontalFlip(p=0.5)
hflip_img = transform(image=image)['image']
vis_img(hflip_img)


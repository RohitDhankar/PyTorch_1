# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
#from PIL import Image
#import albumentations as alb
import random , cv2 

from utils.utils_logger import setup_logger
#pytorch_logger = setup_logger(module_name='logs_pytorch', folder_name=str('pytorch_logs_dir'))

gray_img_input = cv2.imread('./img_inputs/dog.jpg',cv2.IMREAD_GRAYSCALE)/255
cv2.imshow('Dog_gray_img_input',gray_img_input)
cv2.waitKey(0)
#
cv2.VideoCapture()

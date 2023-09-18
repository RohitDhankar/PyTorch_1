"""
For Transforms of the images and augmentation of the images
https://albumentations.ai/docs/

can also use the - https://github.com/albumentations-team/albumentations_examples
Source - https://github.com/albumentations-team/albumentations_examples/blob/master/notebooks/example.ipynb
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
#vis_img(hflip_img)

alb_transform = alb.Compose([
    alb.CLAHE(),
    alb.RandomRotate90(),
    alb.Transpose(),
    alb.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
    alb.Blur(blur_limit=3),
    alb.OpticalDistortion(),
    alb.GridDistortion(),
    alb.HueSaturationValue(),
])

alb_aug_img = alb_transform(image=image)['image']
vis_img(alb_aug_img)






# Project: mlcomp   Author: lightforever   File: config.py    License: Apache License 2.0	5 votes	vote down vote up

# def parse_albu_short(config, always_apply=False):
#     if isinstance(config, str):
#         if config == 'hflip':
#             return A.HorizontalFlip(always_apply=always_apply)
#         if config == 'vflip':
#             return A.VerticalFlip(always_apply=always_apply)
#         if config == 'transpose':
#             return A.Transpose(always_apply=always_apply)

#         raise Exception(f'Unknwon augmentation {config}')
#     assert type(config) == dict
#     return parse_albu([config]) 


#########
# def __init__(
#         self,
#         input_key: str = "image",
#         output_key: str = "rotation_factor",
#         targets_key: str = None,
#         rotate_probability: float = 1.0,
#         hflip_probability: float = 0.5,
#         one_hot_classes: int = None,
#     ):
#         """
#         Args:
#             input_key (str): input key to use from annotation dict
#             output_key (str): output key to use to store the result
#         """
#         self.input_key = input_key
#         self.output_key = output_key
#         self.targets_key = targets_key
#         self.rotate_probability = rotate_probability
#         self.hflip_probability = hflip_probability
#         self.rotate = albu.RandomRotate90()
#         self.hflip = albu.HorizontalFlip()
#         self.one_hot_classes = (
#             one_hot_classes * 8 if one_hot_classes is not None else None
#         ) 

##### FOOBAR 
# oject: kaggle-understanding-clouds   Author: pudae   File: cloud_transform.py    License: BSD 2-Clause "Simplified" License	7 votes	vote down vote up

# def get_training_augmentation(resize_to=(320,640), crop_size=(288,576)):
#     print('[get_training_augmentation] crop_size:', crop_size, ', resize_to:', resize_to) 

#     train_transform = [
#         albu.HorizontalFlip(p=0.5),
#         albu.VerticalFlip(p=0.5),
#         albu.ShiftScaleRotate(scale_limit=0.20, rotate_limit=10, shift_limit=0.1, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
#         albu.GridDistortion(p=0.5),
#         albu.Resize(*resize_to),
#         albu.RandomCrop(*crop_size),
#         albu.ChannelShuffle(),
#         albu.InvertImg(),
#         albu.ToGray(),
#         albu.Normalize(),
#     ]

#     return albu.Compose(train_transform) 



#### FOOBAR 
# roject: albumentations   Author: albumentations-team   File: test_serialization.py    License: MIT License	6 votes	vote down vote up

# def test_transform_pipeline_serialization(seed, image, mask):
#     aug = A.Compose(
#         [
#             A.OneOrOther(
#                 A.Compose(
#                     [
#                         A.Resize(1024, 1024),
#                         A.RandomSizedCrop(min_max_height=(256, 1024), height=512, width=512, p=1),
#                         A.OneOf(
#                             [
#                                 A.RandomSizedCrop(min_max_height=(256, 512), height=384, width=384, p=0.5),
#                                 A.RandomSizedCrop(min_max_height=(256, 512), height=512, width=512, p=0.5),
#                             ]
#                         ),
#                     ]
#                 ),
#                 A.Compose(
#                     [
#                         A.Resize(1024, 1024),
#                         A.RandomSizedCrop(min_max_height=(256, 1025), height=256, width=256, p=1),
#                         A.OneOf([A.HueSaturationValue(p=0.5), A.RGBShift(p=0.7)], p=1),
#                     ]
#                 ),
#             ),
#             A.HorizontalFlip(p=1),
#             A.RandomBrightnessContrast(p=0.5),
#         ]
#     )
#     serialized_aug = A.to_dict(aug)
#     deserialized_aug = A.from_dict(serialized_aug)
#     set_seed(seed)
#     aug_data = aug(image=image, mask=mask)
#     set_seed(seed)
#     deserialized_aug_data = deserialized_aug(image=image, mask=mask)
#     assert np.array_equal(aug_data["image"], deserialized_aug_data["image"])
#     assert np.array_equal(aug_data["mask"], deserialized_aug_data["mask"])     
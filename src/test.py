# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 18:51:53 2019

@author: Keshik
"""


from imgaug import augmenters as iaa
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

seq = iaa.Sequential([
    iaa.Emboss(alpha=(0, 1.0), strength=(0, 0.75)), # emboss images
    iaa.OneOf([
        iaa.GaussianBlur((0, 2.0)), # blur images with a sigma between 0 and 3.0
        iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
        iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
    ]),
    
    
    iaa.OneOf([
        # either change the brightness of the whole image (sometimes
        # per channel) or change the brightness of subareas
        iaa.Multiply((0.8, 1.2), per_channel=0.5),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
    ]),
    
    iaa.OneOf([
        iaa.Dropout(p=0.05, per_channel=True),
        iaa.Crop(px=(0, 4)), # crop images from each side by 0 to 16px (randomly chosen)
    ])
])


image = Image.open("image.png")

print(np.asarray(image).shape)
images_aug = seq.augment_image(np.asarray(image))
print(images_aug.shape)
plt.imshow(images_aug)
plt.show()

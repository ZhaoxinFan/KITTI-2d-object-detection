# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 12:47:01 2019

@author: Keshik
"""

from torch.utils.data.dataset import Dataset
import os
import numpy as np
from random import shuffle
from PIL import Image
import matplotlib.pyplot as plt
import utils

class KITTI2D(Dataset):
    
    def __init__(self, image_dir, label_dir, image_transforms=None, target_transforms=None, fraction = 1.0, split_ratio=0.8, train=True):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_transforms = image_transforms
        self.target_transforms = target_transforms
        self.fraction = fraction
        self.split_ratio = split_ratio
        self.train = train
        self.image_filenames = []
        
        self._load_filenames()
    
    
    def __getitem__(self, index):
        # Returns img_path, img(as PIL), bbox (as np array), labels (as np array)
        bbox, labels = self._read_label(index)
        image = self._read_image(index)
        
        if self.image_transforms is not None:
            image, bbox, labels = self.image_transforms(image, bbox, labels)
        
        if self.target_transforms is not None:
            bbox, labels = self.target_transforms(bbox, labels)
            
        return self._get_img_path(index), image, bbox, labels
    
    
    
    def __len__(self):
        return len(self.image_filenames)
    
    
    def _get_img_path(self, index):
        return "{}/{}".format(self.image_dir, self.image_filenames[index])
    
    def _read_image(self, index):
        # read the file and return the label
        img_path = self._get_img_path(index)
        image = Image.open(img_path)
        
        # Convert grayscale images to rgb
        if (image.mode != "RGB"):
            image = image.convert(mode = "RGB")
            
        return np.array(image)
    
    
    def _read_label(self, index):
        image_filename = self.image_filenames[index]
        label_filename = self._get_label_filename(image_filename)
        
        labels, bounding_boxes = [], []
        lines = []
        
        # Read all lines and close file
        with open(label_filename) as fp:
            lines = fp.readlines()
        fp.close()
        
        # Get labels and bounding boxes
        for objects in lines:
            description = objects.split(" ")
            
            if description[0] not in ["Car", "Pedestrian", "Cyclist", "Person_sitting", "Van"]:
                continue
            
            labels.append(description[0])
            bounding_boxes.append(description[4:8])
        
        #print(labels)
        return np.asarray(bounding_boxes).astype(np.float32), np.asarray(utils.cat2id(labels)).astype(np.int64)
    
    def _load_filenames(self):
        # Load filenames wth absolute paths
        #self.img_filenames = ["{}/{}".format(self.image_dir, i) for i in os.listdir(self.image_dir)]
        self.image_filenames = os.listdir(self.image_dir)
        self.image_filenames.sort()
        
        # Shuffle and sample dataset
        #shuffle(self.image_filenames)
        self.image_filenames = self.image_filenames[:int(len(self.image_filenames)*self.fraction)]
        
        # Create splitted dataset
        if self.train:
            self.image_filenames = self.image_filenames[:int(len(self.image_filenames)*self.split_ratio)]
        else:
            self.image_filenames = self.image_filenames[int(len(self.image_filenames)*self.split_ratio):]
        

    def _get_label_filename(self, image_filename):
        # eg:00000.png
        
        image_id = image_filename.split(".")[0]
        return "{}/{}.txt".format(self.label_dir, image_id)
    
    
# Test here
#kitti = KITTI2D("../data/train/images/", "../data/train/labels/", fraction= 1.0, train=True)
#img_path, img, bounding_boxes, labels = kitti.__getitem__(0)
#plt.imshow(img)
#plt.show()
#print(img_path, bounding_boxes, labels)
#
#utils.plot_bounding_box(img_path, labels, bounding_boxes)
        

# Create a dataloader here and test

        
        

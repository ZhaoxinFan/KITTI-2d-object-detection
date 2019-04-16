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
from skimage.transform import resize
import torch

class KITTI2D(Dataset):
    
    def __init__(self, image_dir, 
                 label_dir,
                 image_size = (416, 416),
                 max_objects = 50,
                 image_transforms=None, 
                 target_transforms=None, 
                 fraction = 1.0, 
                 split_ratio=0.8, 
                 train=True):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.image_transforms = image_transforms
        self.target_transforms = target_transforms
        self.fraction = fraction
        self.split_ratio = split_ratio
        self.train = train
        self.image_filenames = []
        self.max_objects = max_objects
        self.state_variables = {"w":0, "h":0, "pad": (0, 0), "padded_h":0, "padded_w":0 }
        self._load_filenames()
        
    
    
    def __getitem__(self, index):
        # Returns img_path, img(as PIL), bbox (as np array), labels (as np array)
        image = self._read_image(index)
        label = self._read_label(index)
            
        return self._get_img_path(index), image, label
    
    
    def __len__(self):
        return len(self.image_filenames)
    
    
    def _get_img_path(self, index):
        return "{}/{}".format(self.image_dir, self.image_filenames[index])
    
    
    def _read_image(self, index, transform_image=False):
        # read the file and return the label
        img_path = self._get_img_path(index)
        image = Image.open(img_path)
#        
        # Convert grayscale images to rgb
        if (image.mode != "RGB"):
            image = image.convert(mode = "RGB")            

        return torch.from_numpy(self._pad_resize_image(np.array(image), image_size = self.image_size))
    
    
    def _pad_resize_image(self, image, image_size):
        h, w, _ = image.shape
        dim_diff = np.abs(h - w)
        
        # Upper left padding
        pad1 = dim_diff//2
        
        # lower right padding
        pad2 = 0
        
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        
        new_img = np.pad(image, pad, 'constant', constant_values = 128)/255.0
        padded_h, padded_w, _ = new_img.shape
        
        new_img = resize(new_img, (*image_size, 3), mode='reflect')
        
        # Channels first for torch operations
        new_img = np.transpose(new_img, (2, 0, 1))
        
        
        # modify state variables
        self.state_variables["h"] = h
        self.state_variables["w"] = w
        self.state_variables["pad"] = pad
        self.state_variables["padded_h"] = padded_h
        self.state_variables["padded_w"] = padded_w
        
        return new_img
        
    
    def _read_label(self, index):
        image_filename = self.image_filenames[index]
        label_filename = self._get_label_filename(image_filename)
        
        labels = None
        
        if os.path.exists(label_filename):
            labels = np.loadtxt(label_filename).reshape(-1, 5)
            # Access state variables
            w, h, pad, padded_h, padded_w = self.state_variables["w"], self.state_variables["h"], self.state_variables["pad"], self.state_variables["padded_h"], self.state_variables["padded_w"]
            
            # Extract coordinates for unpadded + unscaled image
            x1 = w * (labels[:, 1] - labels[:, 3]/2)
            y1 = h * (labels[:, 2] - labels[:, 4]/2)
            x2 = w * (labels[:, 1] + labels[:, 3]/2)
            y2 = h * (labels[:, 2] + labels[:, 4]/2)
            
            # Adjust for added padding
            x1 += pad[1][0]
            y1 += pad[0][0]
            x2 += pad[1][0]
            y2 += pad[0][0]
            
            # Calculate ratios from coordinates
            labels[:, 1] = ((x1 + x2) / 2) / padded_w
            labels[:, 2] = ((y1 + y2) / 2) / padded_h
            labels[:, 3] *= w / padded_w
            labels[:, 4] *= h / padded_h
            
        
        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 5))
        
        if labels is not None:
            filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        
        filled_labels = torch.from_numpy(filled_labels)

        return filled_labels
            
            
    
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
#kitti = KITTI2D("../data/train/images/", "../data/train/yolo_labels/", fraction= 1.0, train=True)
#img_path, img, labels = kitti.__getitem__(4000)
#print(img_path, img, labels)
#plt.imshow(img.permute(1, 2, 0))
#plt.show()



        
        

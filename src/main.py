# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 16:33:09 2019

@author: user
"""

from __future__ import division

from model import *
from utils import *
from dataset import KITTI2D
from parse_config import *
import os
import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from train_model import train_model
import csv

def main():
    # Set up env variables and create required directories
    os.makedirs("../output", exist_ok=True)
    os.makedirs("../checkpoints", exist_ok=True)
    classes = load_classes("../data/names.txt")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    
    # Set up hyperparameters
    hyperparams = parse_model_config("../config/yolov3-kitti.cfg")[0]
    learning_rate = float(hyperparams["learning_rate"])
    momentum = float(hyperparams["momentum"])
    decay = float(hyperparams["decay"])
    burn_in = int(hyperparams["burn_in"])
    
    
    # File directories
    train_path = "../data/train/images/"
    test_path = "../data/train/images/"
    num_classes = 8
    
    # Create model and load pretrained darknet weights
    model = Darknet("../config/yolov3-kitti.cfg")
    print("Loading imagenet weights to darknet")
    model.load_weights("../checkpoints/darknet53.conv.74")
    model.to(device)
    print(model)
    
    # Create datasets
    train_dataset = KITTI2D("../data/train/images/", "../data/train/yolo_labels/", fraction= 0.05, train=True)
    valid_dataset = KITTI2D("../data/train/images/", "../data/train/yolo_labels/", fraction= 0.05, train=False)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=2, shuffle=False)
    
    # Create optimizers
    print(learning_rate)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate/10.0)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)
    
    
    train_log_file, valid_log_file = open("train.csv","w", newline=""), open("valid.csv","w", newline="")
    train_csv = csv.writer(train_log_file)
    valid_csv = csv.writer(valid_log_file)
    
#    # Train model
    train_model(model, 
          device, 
          optimizer, 
          lr_scheduler, 
          train_dataloader,
          valid_dataloader,
          train_csv,
          valid_csv,
          max_epochs = 2,
          tensor_type=torch.cuda.FloatTensor,
          update_gradient_samples = 16, 
          freeze_darknet=False)
    
    
    train_log_file.close()
    valid_log_file.close()
    
    
    
main()
    
    
    
    
    
    

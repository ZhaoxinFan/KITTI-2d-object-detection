# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 16:33:09 2019

@author: user
"""

from __future__ import division

from model import Darknet
from utils import load_classes
from dataset import KITTI2D
from parse_config import parse_model_config
import os
import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from train_model import train_model
import csv
import warnings
warnings.filterwarnings("ignore")

def main():
    
    # File directories
    train_path = "../data/train/images/"
    test_path = "../data/train/images/"
    labels_path = "../data/train/yolo_labels/"
    weights_path = "../checkpoints/"
    preload_weights_file = "darknet53.conv.74"
    output_path = "../output"
    checkpoints_path = "../checkpoints"
    #class_names_txt = "../data/names.txt"
    #num_classes = 8
    
    
    # Set up env variables and create required directories
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(checkpoints_path, exist_ok=True)
    #classes = load_classes(class_names_txt)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    
    # Set up hyperparameters
    hyperparams = parse_model_config("../config/yolov3-kitti.cfg")[0]
    learning_rate = float(hyperparams["learning_rate"])
    momentum = float(hyperparams["momentum"])
    decay = float(hyperparams["decay"])
    burn_in = int(hyperparams["burn_in"])
    
    
    # Create model and load pretrained darknet weights
    model = Darknet("../config/yolov3-kitti.cfg")
    print("Loading imagenet weights to darknet")
    model.load_weights(os.path.join(weights_path, preload_weights_file))
    model.to(device)
    print(model)
    
    # Create datasets
    train_dataset = KITTI2D(train_path, labels_path, fraction= 0.01, train=True)
    valid_dataset = KITTI2D(test_path, labels_path, fraction= 0.01, train=False)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=2, shuffle=False)
    
    # Create optimizers
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate*5, weight_decay = decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)
    
    # Create log csv files
    train_log_file = open(os.path.join(output_path, "train_results.csv"),"w", newline="")
    valid_log_file = open(os.path.join(output_path, "valid_results.csv"),"w", newline="")
    train_csv = csv.writer(train_log_file)
    valid_csv = csv.writer(valid_log_file)
    
    print("Starting to train model...")
    
    # Train model
    train_model(model, 
          device, 
          optimizer, 
          lr_scheduler, 
          train_dataloader,
          valid_dataloader,
          train_csv,
          valid_csv,
          weights_path,
          max_epochs = 2,
          tensor_type=torch.cuda.FloatTensor,
          update_gradient_samples = 16, 
          freeze_darknet=True,
          freeze_epoch = 2)
    
    
    train_log_file.close()
    valid_log_file.close()
    
    print("Success!")
    

if __name__ == "__main__":
    main()
    
    
    
    
    
    

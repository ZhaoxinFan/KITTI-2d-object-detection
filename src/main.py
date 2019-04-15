# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 16:33:09 2019

@author: user
"""

import os
import logging
import sys
import itertools

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from augmentor import TrainAugmentation, TestTransform, MatchPrior
from dataset import KITTI2D
import config
from multiboxloss import MultiboxLoss
from vgg_ssd import create_vgg_ssd
from tqdm import tqdm
import gc

#train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
#target_transform = MatchPrior(config.priors, config.center_variance,
#                                  config.size_variance, 0.5)
#
#kitti = KITTI2D("../data/train/images/", "../data/train/labels/", fraction= 1.0, 
#                train=True, 
#                image_transforms=train_transform,
#                target_transforms = target_transform)
#img_path, img, bounding_boxes, labels = kitti.__getitem__(0)
#
#print(img_path, img.shape, bounding_boxes.size(), labels.size())
#print(bounding_boxes[:10])



def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    
    
    for img_paths, images, boxes, labels in tqdm(loader):
        #img_paths, images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        confidence, locations = net(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        
        #if i and i % debug_steps == 0:
        
        del images, boxes, labels, img_paths
        torch.cuda.empty_cache()
        gc.collect()
    
    num_samples = len(loader.dataset)
    avg_loss = running_loss / num_samples
    avg_reg_loss = running_regression_loss / num_samples
    avg_clf_loss = running_classification_loss / num_samples
    print("Avg_loss = {}, Avg_reg_loss = {}, Avg_clf_loss = {}".format(avg_loss, avg_reg_loss, avg_clf_loss))
#    logging.info(
#        f"Epoch: {epoch}, Step: {i}, " +
#        f"Average Loss: {avg_loss:.4f}, " +
#        f"Average Regression Loss {avg_reg_loss:.4f}, " +
#        f"Average Classification Loss: {avg_clf_loss:.4f}"
#    )

        

def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0

    for img_paths, images, boxes, labels in tqdm(loader):
        #img_paths, images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    
    num_samples = len(loader.dataset)
    avg_loss = running_loss / num_samples
    avg_reg_loss = running_regression_loss / num_samples
    avg_clf_loss = running_classification_loss / num_samples
    print("Avg_loss = {}, Avg_reg_loss = {}, Avg_clf_loss = {}".format(avg_loss, avg_reg_loss, avg_clf_loss))
    return running_loss / num_samples, running_regression_loss / num_samples, running_classification_loss / num_samples


def main():
    # Create the model first
    net = create_vgg_ssd(num_classes=4)
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)
    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

    kitti_train = KITTI2D("../data/train/images/", "../data/train/labels/", fraction= 0.5, 
                train=True, 
                image_transforms=train_transform,
                target_transforms = target_transform)
    
    kitti_valid = KITTI2D("../data/train/images/", "../data/train/labels/", fraction= 0.5, 
                train=False, 
                image_transforms=test_transform,
                target_transforms = target_transform)
    
    
    train_loader = DataLoader(kitti_train, 16,
                              num_workers=0,
                              shuffle=True)
    
    valid_loader = DataLoader(kitti_valid, 16,
                              num_workers=0,
                              shuffle=True)
    print(len(train_loader.dataset), len(valid_loader.dataset))
    params = [
            {'params': net.base_net.parameters(), 'lr': 0.001},
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': 0.01},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]
    
    use_cuda = True
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")
    print(DEVICE)
    net.to(DEVICE)
    
    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)
    
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9,
                                weight_decay=0.001)
    
    for epoch in range(10):
        train(train_loader, net, criterion, optimizer,
              device=DEVICE, debug_steps=1)
        
        test(valid_loader, net, criterion, DEVICE)
    
    model_path = os.path.join("../models/vgg", "model-2.pth")
    net.save(model_path)

if __name__ == '__main__':
    main()





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
import sys
import time
import datetime
import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import gc


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
#model.apply(weights_init_normal)
print("Loading imagenet weights to darknet")
model.load_weights("../checkpoints/darknet53.conv.74")
#model.load_weights("../checkpoints/kitti.pth")
#model.save_weights("../checkpoints/tkitti.pth")
#model.load_state_dict(torch.load(os.path.join("../checkpoints/", "kitti.pth")))
# move model to device
model.to(device)
#print(model)

# Create datasets
train_dataset = KITTI2D("../data/train/images/", "../data/train/yolo_labels/", fraction= 1.0, train=True)
valid_dataset = KITTI2D("../data/train/images/", "../data/train/yolo_labels/", fraction= 1.0, train=False)

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=2, shuffle=False)

# Create optimizers
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate/10.0)

# Define monitoring params
losses_x = losses_y = losses_w = losses_h = losses_conf = losses_cls = losses_recall = losses_precision = batch_loss= 0.0
update_gradient_batch = 4
best_mAP = 0.0
checkpoint_interval = 1


# Start training here
Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
freeze_backbone = True

for epoch in range(5):
    model.train(True)
    losses_x = losses_y = losses_w = losses_h = losses_conf = losses_cls = losses_recall = losses_precision = batch_loss= 0.0
    
    if freeze_backbone:
        if epoch < 20:
            for i, (name, p) in enumerate(model.named_parameters()):
                if int(name.split('.')[1]) < 75:  # if layer < 75
                    p.requires_grad = False
        elif epoch >= 20:
            for i, (name, p) in enumerate(model.named_parameters()):
                if int(name.split('.')[1]) < 75:  # if layer < 75
                    p.requires_grad = True
                    
    # set gradients to zero
    optimizer.zero_grad()
    
    for i, (img_paths, images, labels) in enumerate(tqdm.tqdm(train_dataloader)):
        images = Variable(images.type(Tensor))
        labels = Variable(labels.type(Tensor), requires_grad=False)
        
        # Calculate loss
        loss = model(images, labels)
        
        # Backpropate
        loss.backward()
        
        # Accumulate losses for plotting later
        losses_x += model.losses["x"]
        losses_y += model.losses["y"]
        losses_w += model.losses["w"]
        losses_h += model.losses["h"]
        losses_conf += model.losses["conf"]
        losses_cls += model.losses["cls"]
        losses_recall += model.losses["recall"]
        losses_precision += model.losses["precision"]
            
        # Update gradients after some batches
        if ((i + 1)%update_gradient_batch == 0) or (i + 1 == len(train_dataloader)):
            optimizer.step()
            optimizer.zero_grad()
        
        del images, labels
        gc.collect()
        torch.cuda.empty_cache()
    
    
    print(
            "[Epoch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
            % (
                epoch,
                10,
                #len(train_dataloader),
                model.losses["x"],
                model.losses["y"],
                model.losses["w"],
                model.losses["h"],
                model.losses["conf"],
                model.losses["cls"],
                loss.item(),
                model.losses["recall"],
                model.losses["precision"],
            )
        )
    
    
    if (epoch+1) % checkpoint_interval == 0:
        #torch.save(model.state_dict(), os.path.join("../checkpoints/","model.pth"))
        model.save_weights("../checkpoints/tkitti.pth")


    # Validation step
    print("Compute %d Epoch mAP..." % epoch)

    all_detections = []
    all_annotations = []
    
    model.train(False)
    for batch_i, (_, images, labels) in enumerate(tqdm.tqdm(valid_dataloader, desc="Detecting objects")):

        images = Variable(images.type(Tensor))

        with torch.no_grad():
            outputs = model(images)
            outputs = non_max_suppression(outputs, 80, conf_thres=0.8, nms_thres=0.4)
            
        for output, annotations in zip(outputs, labels):

            all_detections.append([np.array([]) for _ in range(num_classes)])
            if output is not None:
                # Get predicted boxes, confidence scores and labels
                pred_boxes = output[:, :5].cpu().numpy()
                scores = output[:, 4].cpu().numpy()
                pred_labels = output[:, -1].cpu().numpy()

                # Order by confidence
                sort_i = np.argsort(scores)
                pred_labels = pred_labels[sort_i]
                pred_boxes = pred_boxes[sort_i]

                for label in range(num_classes):
                    all_detections[-1][label] = pred_boxes[pred_labels == label]

            all_annotations.append([np.array([]) for _ in range(num_classes)])
            if any(annotations[:, -1] > 0):

                annotation_labels = annotations[annotations[:, -1] > 0, 0].numpy()
                _annotation_boxes = annotations[annotations[:, -1] > 0, 1:]

                # Reformat to x1, y1, x2, y2 and rescale to image dimensions
                annotation_boxes = np.empty_like(_annotation_boxes)
                annotation_boxes[:, 0] = _annotation_boxes[:, 0] - _annotation_boxes[:, 2] / 2
                annotation_boxes[:, 1] = _annotation_boxes[:, 1] - _annotation_boxes[:, 3] / 2
                annotation_boxes[:, 2] = _annotation_boxes[:, 0] + _annotation_boxes[:, 2] / 2
                annotation_boxes[:, 3] = _annotation_boxes[:, 1] + _annotation_boxes[:, 3] / 2
                annotation_boxes *= 416

                for label in range(num_classes):
                    all_annotations[-1][label] = annotation_boxes[annotation_labels == label, :]
        
        del images, labels
        gc.collect()
        torch.cuda.empty_cache()

    average_precisions = {}
    for label in range(num_classes):
        true_positives = []
        scores = []
        num_annotations = 0

        for i in tqdm.tqdm(range(len(all_annotations)), desc=f"Computing AP for class '{label}'"):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]

            num_annotations += annotations.shape[0]
            detected_annotations = []

            for *bbox, score in detections:
                scores.append(score)

                if annotations.shape[0] == 0:
                    true_positives.append(0)
                    continue

                overlaps = bbox_iou_numpy(np.expand_dims(bbox, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= 0.5 and assigned_annotation not in detected_annotations:
                    true_positives.append(1)
                    detected_annotations.append(assigned_annotation)
                else:
                    true_positives.append(0)

        # no annotations -> AP for this class is 0
        if num_annotations == 0:
            average_precisions[label] = 0
            continue

        true_positives = np.array(true_positives)
        false_positives = np.ones_like(true_positives) - true_positives
        # sort by score
        indices = np.argsort(-np.array(scores))
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = compute_ap(recall, precision)
        average_precisions[label] = average_precision

    print("Average Precisions:")
    mAP = np.mean(list(average_precisions.values()))
    print(f"mAP: {mAP}")





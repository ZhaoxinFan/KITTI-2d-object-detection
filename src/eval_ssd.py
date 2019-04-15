# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 02:19:57 2019

@author: Keshik
"""

import torch
from vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor

import box_utils, measurements
import numpy as np

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from augmentor import TrainAugmentation, TestTransform, MatchPrior, PredictionTransform
from dataset import KITTI2D
import config
from multiboxloss import MultiboxLoss
from tqdm import tqdm
import gc


def group_annotation_by_class(dataset):
    true_case_stat = {}
    all_gt_boxes = {}
    all_difficult_cases = {}
    for i in range(len(dataset)):
        image_id, annotation = dataset._get_annotation(i)
        gt_boxes, classes, is_difficult = annotation
        gt_boxes = torch.from_numpy(gt_boxes)
        for i, difficult in enumerate(is_difficult):
            class_index = int(classes[i])
            gt_box = gt_boxes[i]
            if not difficult:
                true_case_stat[class_index] = true_case_stat.get(class_index, 0) + 1

            if class_index not in all_gt_boxes:
                all_gt_boxes[class_index] = {}
            if image_id not in all_gt_boxes[class_index]:
                all_gt_boxes[class_index][image_id] = []
            all_gt_boxes[class_index][image_id].append(gt_box)
            if class_index not in all_difficult_cases:
                all_difficult_cases[class_index]={}
            if image_id not in all_difficult_cases[class_index]:
                all_difficult_cases[class_index][image_id] = []
            all_difficult_cases[class_index][image_id].append(difficult)

    for class_index in all_gt_boxes:
        for image_id in all_gt_boxes[class_index]:
            all_gt_boxes[class_index][image_id] = torch.stack(all_gt_boxes[class_index][image_id])
    for class_index in all_difficult_cases:
        for image_id in all_difficult_cases[class_index]:
            all_gt_boxes[class_index][image_id] = torch.tensor(all_gt_boxes[class_index][image_id])
    return true_case_stat, all_gt_boxes, all_difficult_cases


def compute_average_precision_per_class(num_true_cases, gt_boxes, difficult_cases,
                                        prediction_file, iou_threshold, use_2007_metric):
    with open(prediction_file) as f:
        image_ids = []
        boxes = []
        scores = []
        for line in f:
            t = line.rstrip().split(" ")
            image_ids.append(t[0])
            scores.append(float(t[1]))
            box = torch.tensor([float(v) for v in t[2:]]).unsqueeze(0)
            #box -= 1.0  # convert to python format where indexes start from 0
            boxes.append(box)
        scores = np.array(scores)
        sorted_indexes = np.argsort(-scores)
        boxes = [boxes[i] for i in sorted_indexes]
        image_ids = [image_ids[i] for i in sorted_indexes]
        true_positive = np.zeros(len(image_ids))
        false_positive = np.zeros(len(image_ids))
        matched = set()
        for i, image_id in enumerate(image_ids):
            box = boxes[i]
            if image_id not in gt_boxes:
                false_positive[i] = 1
                continue

            gt_box = gt_boxes[image_id]
            ious = box_utils.iou_of(box, gt_box)
            max_iou = torch.max(ious).item()
            max_arg = torch.argmax(ious).item()
            if max_iou > iou_threshold:
                if difficult_cases[image_id][max_arg] == 0:
                    if (image_id, max_arg) not in matched:
                        true_positive[i] = 1
                        matched.add((image_id, max_arg))
                    else:
                        false_positive[i] = 1
            else:
                false_positive[i] = 1

    true_positive = true_positive.cumsum()
    false_positive = false_positive.cumsum()
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / num_true_cases
    if use_2007_metric:
        return measurements.compute_voc2007_average_precision(precision, recall)
    else:
        return measurements.compute_average_precision(precision, recall)



def testhere():
    use_cuda=True
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")
    net = create_vgg_ssd(num_classes=4)
    net.load("../models/vgg/model-1.pth")
    predictor = create_vgg_ssd_predictor(net, True, device=DEVICE)
    
    
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)
        
    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

    dataset = KITTI2D("../data/train/images/", "../data/train/labels/", fraction= 0.3, 
                train=False, 
                #image_transforms=test_transform,
                target_transforms = target_transform)
    
    
    
    class_names = [ "Background",
                    "Car",
                   "Pedestrian",
                   "Cyclist"]
    results = []
    
    for i in range(len(dataset)):
        print("process image", i)
        #timer.start("Load Image")
        image = dataset._read_image(i)
        #print("Load Image: {:4f} seconds.".format(timer.end("Load Image")))
        #timer.start("Predict")
        boxes, labels, probs = predictor.predict(image)
        #print("Prediction: {:4f} seconds.".format(timer.end("Predict")))
        
        #print(boxes, labels, probs)
        
        indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i
        results.append(torch.cat([
            indexes.reshape(-1, 1),
            labels.reshape(-1, 1).float(),
            probs.reshape(-1, 1),
            boxes
        ], dim=1))
    results = torch.cat(results)
    
    
    for class_index, class_name in enumerate(class_names):
        if class_index == 0: 
            continue  # ignore background
        
        prediction_path = "det_test_{}.txt".format(class_name)
        with open(prediction_path, "w") as f:
            sub = results[results[:, 1] == class_index, :]
            for i in range(sub.size(0)):
                prob_box = sub[i, 2:].numpy()
                image_id = dataset.image_filenames[int(sub[i, 0])]
                print(
                    image_id + " " + " ".join([str(v) for v in prob_box]),
                    file=f
                )
    
    true_case_stat, all_gb_boxes, all_difficult_cases = group_annotation_by_class(dataset)
    
    aps = []
    print("\n\nAverage Precision Per-class:")
    for class_index, class_name in enumerate(class_names):
        if class_index == 0:
            continue
        prediction_path = "det_test_{}.txt".format(class_name)
        ap = compute_average_precision_per_class(
            true_case_stat[class_index],
            all_gb_boxes[class_index],
            all_difficult_cases[class_index],
            prediction_path,
            0.5,
            True
        )
        aps.append(ap)
        print(f"{class_name}: {ap}")

    print(f"\nAverage Precision Across All Classes:{sum(aps)/len(aps)}")
                
                



testhere()








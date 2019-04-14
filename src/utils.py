# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 14:46:27 2019

@author: Keshik
"""

import cv2
import matplotlib.pyplot as plt

cat2id_dict = { "Background": 0,
                    "Car" : 1, "Van" : 1,
                   "Pedestrian" : 2, "Person_sitting" : 2, 
                   "Cyclist" : 3}

id2cat_dict = {0: "Background", 1: "Car", 2: "Pedestrian", 3: "Cyclist"}

def cat2id(cat_list):
    return [cat2id_dict[i] for i in cat_list]


def id2cat(img_id):
    return id2cat_dict[img_id]
    
   
def plot_bounding_box(image_path, labels, coordinates):
    
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    for i in range(len(labels)):
        label = labels[i]
        bbox = coordinates[i]
        top_left, bottom_right = tuple(bbox[:2]), tuple(bbox[2:])
        #print("id2cat", id2cat(label))
        cv2.rectangle(image, top_left, bottom_right, (0,255,0),1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, id2cat(label) , top_left, font, 0.5, (255,255,255), 1, cv2.LINE_AA)
    
    plt.imshow(image)
    cv2.imwrite("test.png", image, [int(cv2.IMWRITE_PNG_STRATEGY_FIXED), 100])
    
    return image
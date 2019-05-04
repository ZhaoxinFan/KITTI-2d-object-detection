# KITTI 2D Object Detection

## Problem Statement
The goal of this project is to detect object from a number of visual object classes in realistic scenes. There are 7 object classes:
- Car, Van, Truck, Tram
- Pedestrian, Person
- Cyclist

## Data
The label data provided in the KITTI dataset corresponding to a particular image includes the following fields. The labels also include 3D data which is out of scope for this paper.


| Key       	| Values 	| Description                                                                                                           	|
|-----------	|--------	|-----------------------------------------------------------------------------------------------------------------------	|
| type      	| 1      	| String describing the type of object: [Car, Van, Truck, Pedestrian,Person_sitting, Cyclist, Tram, Misc or DontCare]   	|
| truncated 	| 1      	| Float from 0 (non-truncated) to 1 (truncated), where truncated refers to the object leaving image boundaries          	|
| occluded  	| 1      	| Integer (0,1,2,3) indicating occlusion state:  0 = fully visible 1 = partly occluded 2 = largely occluded 3 = unknown 	|
| alpha     	| 1      	| Observation angle of object ranging from [-pi, pi]                                                                    	|
| bbox      	| 4      	| 2D bounding box of object in the image (0-based index): contains left, top, right, bottom pixel coordinates           	|


## Data Augmentations
Since the only has 7481 labelled images, it is essential to incorporate data augmentations to create more variability in available data. The following list provides the types of image augmentations performed.
- Image Embossing
- Blur (Gaussian, Average, Median)
- Brightness variation with per-channel probability
- Adding Gaussian Noise with per-channel probability
- Random dropout of pixels

Geometric augmentations are thus hard to perform since it requires modification of every bounding box coordinate and results in changing the aspect ratio of images. I plan to implement Geometric augmentations in the next release. Examples of image embossing, brightness/ color jitter and Dropout are shown below.

![alt text](./readme_resources/augmentations_final.png)

**Adding Label Noise**  
To allow adding noise to our labels to make the model robust, I performed side by side of cropping images where the number of pixels were chosen from a uniform distribution of [-5px, 5px] where values less than 0 correspond to no crop.

## Data splits
We used an 80 / 20 split for train and validation sets respectively since a separate test set is provided.

## Evaluation Metrics
We use mean average precision (mAP) as the performance metric here.  
**Average Precision:** It is the average precision over multiple IoU values.  
**mAP:** It is average of AP over all the object categories.  




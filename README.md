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
Since the only has 7481 labelled images, it is essential to incorporate data augmentations to create more variability in available data.

Image Augmentations
The following list provides the types of image augmentations we performed.
- Image Embossing
- Blur (Gaussian, Average, Median)
- Brightness variation with per-channel probability
- Adding Gaussian Noise with per-channel probability
- Random dropout of pixels

Geometric augmentations are thus hard to perform since it requires modification of every bounding box coordinate and results in changing the aspect ratio of images. 


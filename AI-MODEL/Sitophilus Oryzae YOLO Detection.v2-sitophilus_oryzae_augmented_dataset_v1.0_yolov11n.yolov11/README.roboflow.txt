
Sitophilus Oryzae YOLO Detection - v2 Sitophilus_Oryzae_Augmented_Dataset_V1.0_YOLOv11n
==============================

This dataset was exported via roboflow.com on June 24, 2026 at 11:56 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 6434 images.
Sitophilus-Oryzae-YOLO-Detection are annotated in YOLOv11 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)
* Auto-contrast via adaptive equalization

The following augmentation was applied to create 10 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise, upside-down
* Randomly crop between 0 and 10 percent of the image
* Random brigthness adjustment of between -20 and +20 percent
* Random exposure adjustment of between -15 and +15 percent
* Random Gaussian blur of between 0 and 1.5 pixels
* Salt and pepper noise was applied to 0.1 percent of pixels



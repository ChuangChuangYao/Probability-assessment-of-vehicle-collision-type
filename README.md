Probability-assessment-of-vehicle-collision-type
=
The model consists of two parts: object detection and tracking with YOLOv4 and DeepSORT methods, and collision type classification based on Convolutional Neural Network (CNN) and Gated Recurrent Unit (GRU), where CNN is used to extract spatial features of the video frames, and GRU is used to train with dynamic features and spatial features to extract the temporal features.

### YOLOv4-DeepSORT
* In YOLO, Each grid cell predict bounding boxes and confidence scores, including five predicted parameters (x, y, w, h, and confidence). `(x, y)` represents the center coordinates of an object in the grid cell, `(w, h)` represents the width and height of the object respectively. The `confidence` indicates the probability of an object, which means Intersection over Union (IoU) between the predicted box and any ground truth box.<br>
* The code of YOLOv4 -DeepSORT is refers to [theAIGuys](https://github.com/theAIGuysCode/yolov4-deepsort). The `object_tracker` was modified into `object_tracker_v1` to output recognized videos and data of bounding box. The allowed classes in `object_tracker_v1` include person, bicycle, car, motorbike, bus, truck. You can clone the YOLOv4-DeepSORT code and execute with raw videos in a file in `object_tracker_exe`.

### Convolutional Neural Network (CNN)
* A Convolutional Neural Network (CNN) can scapture the spatial and temporal dependencies in the image and is developed primarily for image classification. With CNN, the neurons in a three-dimensional structure can be arranged: width, height, and depth, that is image width in pixels, image height in pixels, and RGB channels.
* This study imports the ResNet50 pretrained CNN which was trained on more than a million images from the ImageNet database as a starting point to learn a new task.

### Gated Recurrent Unit (GRU)
*  GRU solves the gradient vanishing problem by replacing hidden nodes in traditional RNN with GRU nodes. Each GRU node consists of an update gate and reset gate.

## The Architecture of the Prediction Model

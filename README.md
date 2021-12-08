Probability-assessment-of-vehicle-collision-type
=
The model consists of two parts: object detection and tracking with YOLOv4 and DeepSORT methods, and collision type classification based on Convolutional Neural Network (CNN) and Gated Recurrent Unit (GRU), where CNN is used to extract spatial features of the video frames, and GRU is used to train with dynamic features and spatial features to extract the temporal features.

### YOLOv4-DeepSORT
* In YOLO, Each grid cell predict bounding boxes and confidence scores, including five predicted parameters (x, y, w, h, and confidence). `(x, y)` represents the center coordinates of an object in the grid cell, `(w, h)` represents the width and height of the object respectively. The `confidence` indicates the probability of an object, which means Intersection over Union (IoU) between the predicted box and any ground truth box.<br>
* The code of YOLOv4 -DeepSORT is refers to [theAIGuys](https://github.com/theAIGuysCode/yolov4-deepsort). The `object_tracker` was modified into `object_tracker_v1` to output recognized videos and data of bounding box. The allowed classes in `object_tracker_v1` include person, bicycle, car, motorbike, bus, truck.



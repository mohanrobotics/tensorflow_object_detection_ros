# tensorflow_object_detection_ros
ROS package for tensor flow object detection. 


## Requirements

* Tensorflow 
* open-cv python

It can be installed using 
`pip install tensorflow==1.14.0.` (Change the version in which the inference graph is built)
`pip install opencv-python`

## Models
Place'frozen_inference_graph.pb' and 'label_map.pbtxt' under `data/`

## Steps

1) `cd catkin_ws/src`
2) `git clone https://github.com/Kukanani/vision_msgs.git`
3) `git clone https://github.com/ros-drivers/usb_cam.git`
4) `git clone https://github.com/mohanrobotics/tensorflow_object_detection_ros.git`
5) `cd ~/catkin_ws && catkin_make`
6) `source ~/catkin_ws/devel/setup.bash`
7) `roslaunch https://github.com/mohanrobotics/tensorflow_object_detection_ros.git object_detect.launch`


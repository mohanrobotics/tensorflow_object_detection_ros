#!/usr/bin/env python

PACKAGE = 'tensorflow_object_detection_ros'
NODE = 'detect_objects'

import os
import sys
import numpy as np
import cv2

try:
    import tensorflow as tf
except ImportError:
    print("unable to import TensorFlow. Is it installed?")

import rospy
from std_msgs.msg import String , Header
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image,CompressedImage
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from tensorflow_object_detection_ros.detection_utils import denormalize_box, apply_non_max_suppression, load_image_into_numpy_array, draw_rectangle, put_text, get_label_map_dict



# PATH_TO_FROZEN_GRAPH = '../data/frcnn_inference_graph/frozen_inference_graph.pb'
PATH_TO_FROZEN_GRAPH = os.path.join(os.path.dirname(sys.path[0]),'data','frozen_inference_graph.pb')
PATH_TO_LABELMAP = os.path.join(os.path.dirname(sys.path[0]),'data', 'label_map.pbtxt')

label_map_dict = get_label_map_dict(PATH_TO_LABELMAP)

print(label_map_dict)
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

conf_threshold = 0.5
nms_threshold = 0.45

class Detector:
    def __init__(self):
        self.bridge = CvBridge()

        self.detection_pub = rospy.Publisher("detections", Detection2DArray, queue_size=1)
        self.sess = tf.Session(graph=detection_graph)

        self.image_sub = rospy.Subscriber("image", CompressedImage, self.get_results, queue_size=1, buff_size=2**24)


    def get_results(self,image_data):
        
        objArray = Detection2DArray()
        objArray.detections =[]
        objArray.header = image_data.header

        rospy.loginfo('Predicting image')

        # Loading the image

        # Alternative snippet for loading the image
        # try:
        #   cv_image = self.bridge.imgmsg_to_cv2(image_data, "bgr8")
        # except CvBridgeError as e:
        #   print(e)
        # image=cv2.cvtColor(cv_image,cv2.COLOR_BGR2RGB)
        # image_np = load_image_into_numpy_array(image)

        #### direct conversion to CV2 ####
        np_arr = np.fromstring(image_data.data, np.uint8)
        # image_np = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:


        if image_np.shape[2] != 3:
            image_np = np.broadcast_to(image_np, (image_np.shape[0], image_np.shape[1], 3)).copy()


        # getting the image shape
        image_shape = image_np.shape[:2]

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # Getting the predicitions
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],feed_dict={image_tensor: image_np_expanded}
            )

        # Filtering the boxes based on conf_threshold
        filtered_scores = [scores[0][i] for i in np.where(scores[0] > conf_threshold)]
        filtered_boxes = [boxes[0][i] for i in np.where(scores[0] > conf_threshold)]
        filtered_classes = [classes[0][i] for i in np.where(scores[0] > conf_threshold)]


        display_img = image_np

        if len(filtered_scores[0]) != 0:
            # NMS thresholding
            indices, count = apply_non_max_suppression(filtered_boxes[0], filtered_scores[0], nms_threshold, 200)
            selected_indices = indices[:count]

            ## Getting the final boxes
            final_boxes = filtered_boxes[0][selected_indices]
            final_scores = filtered_scores[0][selected_indices]
            final_classes = filtered_classes[0][selected_indices]


            final_boxes = [denormalize_box(box, image_shape) for box in final_boxes]

            for box, conf , pred_class in zip(final_boxes, final_scores, final_classes) :

                # Box format  y_min, x_min, y_max, x_max 

               #               (x_min,y_min)..................                    #
               #               .                             .                    #
               #               .                             .                    #
               #               .                             .                    #
               #               .                             .                    #
               #               .                             .                    #
               #               .                             .                    #
               #               .                             .                    #
               #               .                             .                    #
               #               ....................(x_max_y_max)                   #


                objArray.detections.append(self.ind_obj_prediction(image_data.header,box,conf,pred_class,image_shape))
                display_img = draw_rectangle(display_img, (box[1], box[0]), (box[3],box[2]),int(pred_class))
                display_img = put_text(display_img, label_map_dict[int(pred_class)], (box[1], box[0]), int(pred_class))


            rospy.loginfo(final_boxes)
            self.detection_pub.publish(objArray)


        cv2.imshow('cv_img', display_img)
        cv2.waitKey(2)


    def ind_obj_prediction(self,header,box,conf,pred_class,image_shape):

        image_height, image_width = image_shape

        obj=Detection2D()
        obj_hypothesis= ObjectHypothesisWithPose()

        obj.header= header
        obj_hypothesis.id = pred_class
        obj_hypothesis.score = conf
        obj.results.append(obj_hypothesis)
        obj.bbox.size_y = int((box[2]-box[0])*image_height)
        obj.bbox.size_x = int((box[3]-box[1] )*image_width)
        obj.bbox.center.x = int((box[1] + box [3])*image_height/2)
        obj.bbox.center.y = int((box[0] + box[2])*image_width/2)

        return obj

if __name__ == '__main__':
    rospy.init_node('detect_objects')
    n = Detector()
    rospy.spin()
    cv2.destroyAllWindows()

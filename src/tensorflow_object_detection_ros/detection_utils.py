#!/usr/bin/env python
import numpy as np
import cv2

GREEN = (0, 255, 0)
LINE = cv2.LINE_AA
FONT = cv2.FONT_HERSHEY_SIMPLEX

STANDARD_COLORS = [ (240,230,140),  (128,128,0),    (255,255,0),
 (154,205,50), (255,0,0), (0,255,0) , (128,128,128),    (0,0,128),
 (0,128,128), (128,0,128) ]

def apply_non_max_suppression(boxes, scores, iou_thresh=.45, top_k=200):
    """Apply non maximum suppression.

    # Arguments
        boxes: Numpy array, box coordinates of shape (num_boxes, 4)
            where each columns corresponds to x_min, y_min, x_max, y_max
        scores: Numpy array, of scores given for each box in 'boxes'
        iou_thresh : float, intersection over union threshold
            for removing boxes.
        top_k: int, number of maximum objects per class

    # Returns
        selected_indices: Numpy array, selected indices of kept boxes.
        num_selected_boxes: int, number of selected boxes.
    """

    selected_indices = np.zeros(shape=len(scores))
    if boxes is None or len(boxes) == 0:
        return selected_indices
    # x_min = boxes[:, 0]
    # y_min = boxes[:, 1]
    # x_max = boxes[:, 2]
    # y_max = boxes[:, 3]
    x_min = boxes[:, 1]
    y_min = boxes[:, 0]
    x_max = boxes[:, 3]
    y_max = boxes[:, 2]

    areas = (x_max - x_min) * (y_max - y_min)
    remaining_sorted_box_indices = np.argsort(scores)
    remaining_sorted_box_indices = remaining_sorted_box_indices[-top_k:]

    num_selected_boxes = 0
    while len(remaining_sorted_box_indices) > 0:
        best_score_args = remaining_sorted_box_indices[-1]
        selected_indices[num_selected_boxes] = best_score_args
        num_selected_boxes = num_selected_boxes + 1
        if len(remaining_sorted_box_indices) == 1:
            break

        remaining_sorted_box_indices = remaining_sorted_box_indices[:-1]

        best_x_min = x_min[best_score_args]
        best_y_min = y_min[best_score_args]
        best_x_max = x_max[best_score_args]
        best_y_max = y_max[best_score_args]

        remaining_x_min = x_min[remaining_sorted_box_indices]
        remaining_y_min = y_min[remaining_sorted_box_indices]
        remaining_x_max = x_max[remaining_sorted_box_indices]
        remaining_y_max = y_max[remaining_sorted_box_indices]

        inner_x_min = np.maximum(remaining_x_min, best_x_min)
        inner_y_min = np.maximum(remaining_y_min, best_y_min)
        inner_x_max = np.minimum(remaining_x_max, best_x_max)
        inner_y_max = np.minimum(remaining_y_max, best_y_max)

        inner_box_widths = inner_x_max - inner_x_min
        inner_box_heights = inner_y_max - inner_y_min

        inner_box_widths = np.maximum(inner_box_widths, 0.0)
        inner_box_heights = np.maximum(inner_box_heights, 0.0)

        intersections = inner_box_widths * inner_box_heights
        remaining_box_areas = areas[remaining_sorted_box_indices]
        best_area = areas[best_score_args]
        unions = remaining_box_areas + best_area - intersections
        intersec_over_union = intersections / unions
        intersec_over_union_mask = intersec_over_union <= iou_thresh
        remaining_sorted_box_indices = remaining_sorted_box_indices[
            intersec_over_union_mask]

    return selected_indices.astype(int), num_selected_boxes

def denormalize_box(box, image_shape):
    """Scales corner box coordinates from normalized values to image dimensions.
    # Arguments
        box: Numpy array containing corner box coordinates.
        image_shape: List of integers with (height, width).
    # Returns
        returns: box corner coordinates in image dimensions
    """
    # x_min, y_min, x_max, y_max = box[:4]
    y_min, x_min, y_max, x_max = box[:4]

    height, width = image_shape
    x_min = int(x_min * width)
    y_min = int(y_min * height)
    x_max = int(x_max * width)
    y_max = int(y_max * height)

    # return [x_min, y_min, x_max, y_max]
    return [y_min, x_min, y_max, x_max]


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    channel_dict = {'L':1, 'RGB':3} # 'L' for Grayscale, 'RGB' : for 3 channel images
    return np.array(image.getdata()).reshape((im_height, im_width, channel_dict[image.mode])).astype(np.uint8)

def draw_rectangle(image, corner_A, corner_B, color_id, thickness=1):
    """ Draws a filled rectangle from corner_A to corner_B.
    # Arguments
        image: Numpy array of shape [H, W, 3].
        corner_A: List/tuple of length two indicating (y,x) openCV coordinates.
        corner_B: List/tuple of length two indicating (y,x) openCV coordinates.
        color: List of length three indicating BGR color of point.
        thickness: Integer/openCV Flag. Thickness of rectangle line.
            or for filled use cv2.FILLED flag.
    """
    color = STANDARD_COLORS[color_id%(len(STANDARD_COLORS))]
    return cv2.rectangle(
        image, tuple(corner_A), tuple(corner_B), tuple(color), thickness)


def put_text(image, text, point, color_id, scale = 0.9, thickness=1):
    """Draws text in image.
    # Arguments
        image: Numpy array.
        text: String. Text to be drawn.
        point: Tuple of coordinates indicating the top corner of the text.
        scale: Float. Scale of text.
        color: Tuple of integers. BGR color coordinates.
        thickness: Integer. Thickness of the lines used for drawing text.
    """
    color = STANDARD_COLORS[color_id%(len(STANDARD_COLORS))]
    return cv2.putText(image, text, point, FONT, scale, color, thickness, LINE)


def get_label_map_dict(label_map_path):
    
    """
    Args:
    label_map_path: path to label_map.
    Returns:
    A dictionary mapping label names to id.
    """
    label_map_dict = {}
    for line in open(label_map_path):
        
        line = line.rstrip('\n')
        if 'id:' in line:
            object_id = int(line.split(':')[-1])
        if 'name:' in line:
            object_name = str(line.split(':')[-1])
            label_map_dict[object_id] = object_name
    return label_map_dict

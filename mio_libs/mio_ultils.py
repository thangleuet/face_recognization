import json
import cv2
import numpy as np
import torch

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def xywh2_cent_xywh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = y[:, 0] + y[:, 2] / 2
    y[:, 1] = y[:, 1] + y[:, 3] / 2
    return y

def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def tranfer_image(img,x1,y1,x2,y2):
    try:
        sub_img = img[y1: y2, x1: x2]
        white_rect = np.ones(sub_img.shape, dtype = np.uint8) * 1
        res = cv2.addWeighted(sub_img, 0.3, white_rect, 0.3, 1.0)
        # Putting the image back to its position
        img[y1: y2, x1: x2] = res
    except Exception as e:
        print(e)
    return img


def cal_boxes_overlap_area(box1, box2):
    """
    Calculate the intersection area of 2 boxes

    Args:
        box1 (list)): [x,y,w,h]
        box2 (list): [x,y,w,h]

    Returns:
        float: intersection area 
    """        

    x1 = max(min(box1[0], box1[0] + box1[2]), min(box2[0], box2[0] + box2[2]))
    y1 = max(min(box1[1], box1[1] + box1[3]), min(box2[1], box2[1] + box2[3]))
    x2 = min(max(box1[0], box1[0] + box1[2]), max(box2[0], box2[0] + box2[2]))
    y2 = min(max(box1[1], box1[1] + box1[3]), max(box2[1], box2[1] + box2[3]))
    if x1<x2 and y1<y2:
        return (x2 - x1) * (y2 - y1)
    return 0

def cal_boxes_iou(box1, box2):
    """Func IOU boxes head and persons

    Args:
        box1 (list)): [x,y,w,h]
        box2 (list): [x,y,w,h]

    Returns:
        float: iou
    """        

    # Get the Width and Height of each bounding box
    width_box1 = box1[2]
    height_box1 = box1[3]
    width_box2 = box2[2]
    height_box2 = box2[3]
    
    # Calculate the area of the each bounding box
    area_box1 = width_box1 * height_box1
    area_box2 = width_box2 * height_box2
    
    # Find the vertical edges of the union of the two bounding boxes
    mx = min(box1[0] - width_box1/2.0, box2[0] - width_box2/2.0)
    Mx = max(box1[0] + width_box1/2.0, box2[0] + width_box2/2.0)
    
    # Calculate the width of the union of the two bounding boxes
    union_width = Mx - mx
    
    # Find the horizontal edges of the union of the two bounding boxes
    my = min(box1[1] - height_box1/2.0, box2[1] - height_box2/2.0)
    My = max(box1[1] + height_box1/2.0, box2[1] + height_box2/2.0)    
    
    # Calculate the height of the union of the two bounding boxes
    union_height = My - my
    
    # Calculate the width and height of the area of intersection of the two bounding boxes
    intersection_width = width_box1 + width_box2 - union_width
    intersection_height = height_box1 + height_box2 - union_height

    # If the the boxes don't overlap then their IOU is zero
    if intersection_width <= 0 or intersection_height <= 0:
        return 0.0

    # Calculate the area of intersection of the two bounding boxes
    intersection_area = intersection_width * intersection_height
    
    # Calculate the area of the union of the two bounding boxes
    union_area = area_box1 + area_box2 - intersection_area
    
    # Calculate the IOU
    iou = intersection_area/union_area
    
    return iou

def is_box_inside_box(x1, y1, w1, h1, x2, y2, w2, h2):
    """check if 1st box is inside 2nd box
    """    
    
    return x1 >= x2 and \
           y1 >= y2 and \
           x1 + w1 <= x2 + w2 and \
           y1 + h1 <= y2 + h2 

def obj_dict(obj):
        return obj.__dict__

class encoder(json.JSONEncoder):
        def default(self, o):
                return o.__dict__
            

def draw_text_with_bg(image, text: str, topleft: tuple, bottomleft: tuple, font: int, font_scale: float, color: tuple, thickness: float, background_color: tuple, backgroud_alpha: float = 1.0):
    """Draw text on image with background color

    Args:
        image (cvImage): Input/Output image to draw
        text (str): Text message
        topleft (tuple): Top-left corner of text. This is used as defaul alignment
        bottomleft (tuple): Bottom-Left corner of text. If it is set, text will be align to the bottom. Value of topleft will be ignored
        font (int): Opencv font 
        font_scale (float): Opencv font scale
        color (tuple): Text color   
        thickness (float): Text thichness
        background_color (tuple): Background color in BGR
        backgroud_alpha (float, optional): Background transparency coefficient. Defaults to 1.0.

    Returns:
        Image: Drawn image
    
    Usage:
         
    # Draw with top-left alignment
    draw_text_with_bg(
        image=img,
        text="Top-Left",
        topleft=box[1],
        bottomleft=None,
        font=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        font_scale=2,
        color=(0, 0, 255),
        thickness=3,
        background_color=(255, 0, 0),
        backgroud_alpha=0.5)
        
    # Draw with bottom-left alignment
    draw_text_with_bg(
        image=img,
        text="Bottom-Right",
        topleft=None,
        bottomleft=box[1],
        font=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        font_scale=2,
        color=(0, 0, 255),
        thickness=3,
        background_color=(255, 0, 0),
        backgroud_alpha=0.5)
    """    
    
    try:
        # Get text size
        (text_w, text_h), text_base_line = cv2.getTextSize(text=text, fontFace=font, fontScale=font_scale,thickness=thickness)
        text_base_line += thickness
        
        # Draw rectangle
        x1, y1 = 0, 0
        if topleft is not None: # Use top-left alignment as default
            x1, y1 = topleft
        else:
            x1, y1 = bottomleft[0], bottomleft[1] - text_h - text_base_line
        
        x2 = x1 + text_w
        y2 = y1 + text_h + text_base_line
        
        if backgroud_alpha > 0.99:
            cv2.rectangle(img=image, pt1 = (x1, y1), pt2= (x2, y2), color= background_color, thickness=-1)
        else:
            # Validate
            img_w, img_h = image.shape[1], image.shape[0]
            x1 = min(max(x1, 0), img_w)
            y1 = min(max(y1, 0), img_h)
            
            x2 = min(max(x2, 0), img_w)
            y2 = min(max(y2, 0), img_h)
            
            if x1 > x2: x1, x2 = x2, x1 # swap
            if y1 > y2: y1, y2 = y2, y1 # swap
            
            if (x2 - x1) * (y2 - y1) > 0:
                sub_img = image[y1: y2, x1: x2]
                white_rect = np.full(sub_img.shape, background_color, dtype=np.uint8)
                image[y1: y2, x1: x2] = cv2.addWeighted(image[y1: y2, x1: x2], backgroud_alpha, white_rect, 1 - backgroud_alpha, 0)
        
        # Put text on
        text_x = x1
        text_y = y1 +  text_h 
        cv2.putText(img=image, text=text, org=(text_x, text_y), fontFace=font, fontScale=font_scale, color=color, thickness=thickness)
        
        return image
    except Exception as e:
        print(e)
    return image
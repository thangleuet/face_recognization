import cv2
from pathlib import Path
from pandas import DataFrame
import sys
# import tensorflow as tf
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras import applications
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def positioning(frame, list_person, track):
    outputs = None
    confs = None
    if len(list_person) > 0:
        list_frame = np.array(list_person)
        list_frame[:, 2] = list_frame[:, 0] + list_frame[:, 2]
        list_frame[:, 3] = list_frame[:, 1] + list_frame[:, 3]
        xywhs = torch.from_numpy(xyxy2xywh(list_frame[:, 0:4]))
        confs = torch.from_numpy(list_frame[:, 4])
        clss = torch.from_numpy(list_frame[:, 5])
        outputs = track.update(xywhs, confs, clss, frame)

    return outputs, confs




def tranfer_image(img,x1,y1,x2,y2):
    sub_img = img[y1: y2, x1: x2]
    white_rect = np.ones(sub_img.shape, dtype = np.uint8) * 1
    res = cv2.addWeighted(sub_img, 0.7, white_rect, 0.7, 1.0)
    # Putting the image back to its position
    img[y1: y2, x1: x2] = res
    return img

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


# if __name__ == "__main__":
#     list_person
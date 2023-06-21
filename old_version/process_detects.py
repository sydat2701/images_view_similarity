# from ultralytics import YOLO
import cv2
import numpy as np
import os
import torch
from PIL import Image
from detect import detect
from yolov5.detect import detect

def get_coor(bbox):
    center_x1, center_y1, center_x2, center_y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    
    #trai tren, phai duoi
    return int(center_x1), int(center_y1), int(center_x2), int(center_y2)
    return int((center_x-width/2)), int((center_y-height/2)), int((center_x+width/2)), int((center_y+height/2))


def get_detected_img(path, conf_thres):

    res = detect(source=path, conf_thres=conf_thres)
    img=cv2.imread(path)

    
    if res is None:
        return img, 0,0, img.shape[1], img.shape[0]
    bbox = res[0]
    
    img_new = np.zeros(img.shape, dtype='uint8')
    t11, t12, t13,t14 = get_coor(bbox)


    img_new[  t12:t14, t11:t13,:] = img[ t12:t14,t11:t13, :]

    return img_new, t11, t12, t13, t14
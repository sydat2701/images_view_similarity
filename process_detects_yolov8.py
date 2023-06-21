# from ultralytics import YOLO
import cv2
import numpy as np
import os
import torch
from PIL import Image
#from detect import detect
from yolov8.ultralytics import YOLO

model = YOLO("yolov8/best.pt") 

def get_coor(bbox):
    center_x1, center_y1, center_x2, center_y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    
    #trai tren, phai duoi
    return int(center_x1), int(center_y1), int(center_x2), int(center_y2)
    return int((center_x-width/2)), int((center_y-height/2)), int((center_x+width/2)), int((center_y+height/2))


def get_detected_img(path, conf_thres):


    #------------------------------------------------------ imgsz = 320 so that detection will outputs the true bounding box with the real image dimensions -------------------------------------
    res = model.predict(path, save=False, imgsz=320, conf=conf_thres)
    img=cv2.imread(path)

    
    if len(res[0].boxes.xyxy) == 0:
        return img, 0,0, img.shape[1], img.shape[0]
    
    #chi lay bbox dau tien -----------------------------------------------------------------------------------------------------------------
    try:
        bbox = res[0].boxes.xyxy[0]
    except:
        print("-------------------------------")
        print(res[0].boxes)
    
    img_new = np.zeros(img.shape, dtype='uint8')
    t11, t12, t13,t14 = get_coor(bbox)
    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # print(path)
    # print(res[0].boxes)
    # print(t11, t12, t13, t14)


    img_new[  t12:t14, t11:t13,:] = img[ t12:t14,t11:t13, :]

    return img_new, t11, t12, t13, t14
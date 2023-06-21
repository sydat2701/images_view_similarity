from yolov5.detect import detect
import cv2
import numpy as np
import os
import torch
from PIL import Image

def get_coor(bbox):
    center_x1, center_y1, center_x2, center_y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    
    #trai tren, phai duoi
    return int(center_x1), int(center_y1), int(center_x2), int(center_y2)
    return int((center_x-width/2)), int((center_y-height/2)), int((center_x+width/2)), int((center_y+height/2))



path2='14.jpg'
path1='14.jpg'

img1 = cv2.imread(path1)


res1 = detect(source=path1)




bbox1 = res1[0]



#print("shape: ", img1.shape)
#img2= cv2.imread(path2)

img1_new = np.zeros(img1.shape, dtype='uint8')
t11, t12, t13,t14 = get_coor(bbox1)
# print(":::::::::::: ", t11,t12,t13,t14)

#tmp=cv2.rectangle(img1, (t11, t12), (t13, t14), color=(0,255,0), thickness=3)
# cv2.imwrite('./abc1.jpg', tmp)


img1_new[  t12:t14, t11:t13,:] = img1[ t12:t14,t11:t13, :]

cv2.imwrite('./abc1.jpg', img1_new)


from ultralytics import YOLO
import cv2
import numpy as np
import os
import torch
from PIL import Image

def get_coor(bbox):
    center_x, center_y, width, height = bbox[0], bbox[1], bbox[2], bbox[3]
    
    #trai tren, phai duoi
    #return int(center_x), int(center_y), int(width), int(height)
    return int((center_x-width/2)), int((center_y-height/2)), int((center_x+width/2)), int((center_y+height/2))

def pre_process(img):
    # img = Image.open(path)
    # img = img.convert('RGB')
    #img = img.transpose(2,0,1)
    #img = np.ascontiguousarray(img)
    img = img[..., ::-1].transpose((2, 0, 1))
    img = torch.from_numpy(img)
    img=img.unsqueeze(0)
    img = img.to('cuda:0')
    img = img.float()
    img /= 255
    return img

# Load a model
model = YOLO("yolov8s.yaml")  # build a new model from scratch
#model = YOLO(f"/home/nts1/users/datts/image_matching/ultralytics-main/runs/detect/train/weights/best.pt")  # load a pretrained model (recommended for training)
model= model.load('/home/nts1/users/datts/image_matching/ultralytics-main/runs/detect/train/weights/best.pt')

#path2='/home/nts1/users/datts/image_matching/ultralytics-main/loftr/1.jpg'
path1='/home/nts1/users/datts/image_matching/ultralytics-main/loftr/2.jpg'

img1 = cv2.imread(path1)
#img1= pre_process(img1)

res1 = model.predict(img1)
#res2 = model.predict(path2)


# print("-----------------------------------")
# print(res[0].boxes.data)
print("====================")
# print(res1)
print(res1[0].boxes.data)
# print(res2[0].boxes.data)

bbox1 = res1[0].boxes.data[0]
#bbox2 = res2[0].boxes.data[0]


print("shape: ", img1.shape)
#img2= cv2.imread(path2)

img1_new = np.zeros((img1.shape[1], img1.shape[0], 3), dtype='uint8')
t11, t12, t13,t14 = get_coor(bbox1)
print(":::::::::::: ", t11,t12,t13,t14)

tmp=cv2.rectangle(img1, (t11, t12), (t13, t14), color=(0,255,0), thickness=3)
cv2.imwrite('./abc1.jpg', tmp)


img1_new[ t12:t14, t11:t13, :] = img1[t12:t14, t11:t13, :]

cv2.imwrite('./abc.jpg', img1_new)

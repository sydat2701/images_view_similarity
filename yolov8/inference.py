import cv2
from ultralytics import YOLO

# Load a model
#model = YOLO("yolov8s.yaml")  # build a new model from scratch
model = YOLO("best.pt")  # load a pretrained model (recommended for training)
#model= model.load('best.pt')
# img=cv2.imread('zidane.jpg')
img=cv2.imread('abc1.jpg')
# res = model(img)
res = model.predict('test', save=True, imgsz=320, conf=0.5)
print("-----------------------------------")
print("+++++++++++++++++++++++")
print(res[0].boxes.data)
print(res[1].boxes.data)
print(res[2].boxes.data)
print(res[3].boxes.data)

# img=cv2.imread('/home/nts1/users/datts/image_matching/ultralytics-main/test_img/14915_2019_09_27_11_41_37image1.jpg')
# print(img.shape)
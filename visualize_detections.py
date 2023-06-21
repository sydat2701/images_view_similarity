from src.loftr import LoFTR, default_cfg
import torch
import cv2
import matplotlib.cm as cm
from src.utils.plotting import make_matching_figure
import numpy as np
from tsai.models.MINIROCKET import load_minirocket
from tsai.all import combine_split_data
from tsai.models.MINIROCKET import MiniRocketClassifier
import os
from yolov5.detect import detect
from process_detects import get_detected_img

data_path = 'test_img/img2'
pair_path = 'pairs/pairs.txt'


threshold =0.5
thresh_num_kpts = 1000


# Initialize LoFTR
matcher = LoFTR(config=default_cfg)
#matcher.load_state_dict(torch.load("weights/indoor_ds_new.ckpt")['state_dict'])
matcher.load_state_dict(torch.load("weights/outdoor_ds.ckpt")['state_dict'])
matcher = matcher.eval().cuda()

def abandon_out_of_roi(mkpts1, mkpts2, mconf, scale1, scale2, x11, y11, x12, y12, x21, y21, x22, y22):

    keeper1 = []
    keeper2 = []


    for point in mkpts1:
        if point is None:
            keeper1.append(1)
            continue
        
        if (point[0]<(x11/scale1[0])) or (point[0]>(x12/scale1[0])) or (point[1]<(y11/scale1[1]) or (point[1]>(y12/scale1[1]))):
            keeper1.append(1)    # bang 1 neu nam ngoai vung detect va bang 0 vice versa
        else:
            keeper1.append(0)
    
    #print("---------------------------------------------------------------------------------------------------------------------------")
    
    for point in mkpts2:

        if (point[0]<(x21/scale2[0])) or (point[0]>(x22/scale2[0])) or (point[1]<(y21/scale2[1])) or (point[1]>(y22/scale2[1])):
            
            keeper2.append(1)    # bang 1 neu nam ngoai vung detect va bang 0 vice versa
        else:
            keeper2.append(0)



    res1, res2, conf = [], [], []
    idx=0
    for i, j in zip(keeper1, keeper2):
        if (i==j) and (i==0):
            res1.append(mkpts1[idx])
            res2.append(mkpts2[idx])
            conf.append(mconf[idx])
        idx +=1
    
    return np.array(res1), np.array(res2), np.array(conf)


num_0 =0
fail = 0

def inference(path1):
    # Load example images
    img0_pth = path1
    # img1_pth = path2
    try:
        dt_img0, x11, y11, x12, y12 = get_detected_img(img0_pth)
        return dt_img0

        

        # img0_raw = cv2.resize(img0_raw, (640, 480))
        # img1_raw = cv2.resize(img1_raw, (640, 480))

    except:
        global fail
        fail +=1
        return -1
    

    
    return 0
        

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from glob import glob

def plot_album(img1, img2, path):
    plt.cla()
    '''fig, axes = plt.subplots(nrows=1, ncols=2)
    # this assumes the images are in images_dir/album_name/<name>.jpg
    for imp, ax in zip(list_img, axes.ravel()):
        #img = mpimg.imread(imp)
        ax.imshow(img)
        ax.axis('off')
    fig.tight_layout()
    plt.save(path)'''

    rows = 1
    columns = 2

    fig = plt.figure(figsize=(50, 50))
    fig.add_subplot(rows, columns, 1)

    
    fig.add_subplot(1, 2, 1)  
    # showing image
    plt.imshow(img1)
    plt.axis('off')
    plt.title("original")  

    fig.add_subplot(1, 2, 2)  
    # showing image
    plt.imshow(img2)
    plt.axis('off')
    plt.title("detection")

    plt.savefig(path)

    



path_img='/home/nts1/users/datts/image_matching/perspective_img'

for fol in os.listdir(path_img):
    if fol=='unchange':
        continue
    fol_path= os.path.join(path_img, fol)
    for img_fol in os.listdir(fol_path):
        img_fol_path = os.path.join(fol_path, img_fol)

        target_fol = os.path.join('detections', fol, img_fol)
        os.makedirs(target_fol)

        for idx, img in enumerate(os.listdir(img_fol_path)):
            img_path = os.path.join(img_fol_path, img)
        

            r = inference(img_path)
            #cv2.imwrite(os.path.join(target_fol, img), r)
            ori_img = cv2.imread(img_path)
            plot_album(ori_img, r, os.path.join(target_fol, img))
        #cv2.imwrite(os.path.join(target_fol, 'original.jpg'), )

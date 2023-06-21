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
import pickle
#from yolov5.detect import detect
from process_detects_yolov8 import get_detected_img
from scipy.special import rel_entr, kl_div

threshold = 0.7
conf_dt =0.5 #confidence of detection

#-------------------------- config ------------------------------------------------
data_path = 'test_img/img8'
pair_path = 'pairs/img8.txt'
kld_thres = 30.5
num_features = 800

#---------------------------------------------------------------------------------



# Initialize LoFTR
matcher = LoFTR(config=default_cfg)
matcher.load_state_dict(torch.load("weights/outdoor_ds.ckpt")['state_dict'])
matcher = matcher.eval().cuda()


def calculate_perspective_dis(arr1, arr2):
    matrix = cv2.getPerspectiveTransform(arr1, arr2)
    I = np.eye(matrix.shape[0])
    return np.sum(np.abs(matrix-I))

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


ttt = 0


def inference(path1, path2, name, conf_dt):
    # Load example images
    img0_pth = path1
    img1_pth = path2



    dt_img0, x11, y11, x12, y12 = get_detected_img(img0_pth, conf_dt)
    img0_raw = cv2.cvtColor(dt_img0, cv2.COLOR_BGR2GRAY)
    img0_shape = dt_img0.shape  #(H, W, C)

    dt_img1, x21, y21, x22, y22 = get_detected_img(img1_pth, conf_dt)
    img1_raw = cv2.cvtColor(dt_img1, cv2.COLOR_BGR2GRAY)
    img1_shape = dt_img1.shape

    global ttt
    ttt +=1 

    cv2.imwrite('tmp/'+str(ttt)+'.jpg', dt_img0)
    cv2.imwrite('tmp/'+str(ttt)+'_1.jpg', dt_img1)

    img0_raw = cv2.resize(img0_raw, (640, 480))
    img1_raw = cv2.resize(img1_raw, (640, 480))

    img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
    img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
    batch = {'image0': img0, 'image1': img1}

    # Inference with LoFTR and get prediction
    with torch.no_grad():
        matcher(batch)
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        mconf = batch['mconf'].cpu().numpy()

        flag = mconf>threshold
        mkpts0 =mkpts0[flag]
        mkpts1=mkpts1[flag]
        mconf = mconf[flag]




        #-------------------------loai bo cac kpts nam ngoai vung detect-------------------------------------------
        scale1= (img0_shape[1]/640, img0_shape[0]/480)
        scale2= (img1_shape[1]/640, img1_shape[0]/480)
        mkpts0, mkpts1, mconf = abandon_out_of_roi(mkpts0, mkpts1, mconf, scale1, scale2, x11, y11, x12, y12, x21, y21, x22, y22)

        #-------------------------------------------------------------------------------------
        tmp_x1 = mkpts0.reshape(-1)
        tmp_x2 = mkpts1.reshape(-1)

       
        tmp_x1=tmp_x1[:num_features*2]
        tmp_x2=tmp_x2[:num_features*2]

        #---------------------------------kld metrics------------------------------------------------------
        kld = sum(abs(rel_entr(tmp_x1, tmp_x2)))/len(tmp_x1)

        '''# Draw
        color = cm.jet(mconf)
        text = [
            'LoFTR',
            'Matches: {}'.format(len(mkpts0)),
        ]
        fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, text=text, 
                                   path='/home/nts1/users/datts/image_matching/LoFTR/res_img/'+name+'_pred'+str(str(avg).split('.')[0]) +'.jpg')'''
        
        if kld > kld_thres:
            return 1 #da thay doi
        else:
            return 0 #khong thay doi
       



f=open(pair_path, "r")
lines=f.readlines()
f.close()

for idx, line in enumerate(lines):
    imgs = line[:-1].split(' ')
    path1, path2 = os.path.join(data_path, imgs[0]), os.path.join(data_path, imgs[1])
    print("------------------------------------------------------------------")
    print(path1, path2)
    res = inference(path1, path2, imgs[0].split('.')[0]+'_'+imgs[1].split('.')[0], conf_dt)
    print("Res: ", res)
    
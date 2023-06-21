from src.loftr import LoFTR, default_cfg
import torch
import cv2
import matplotlib.cm as cm
from src.utils.plotting import make_matching_figure
import numpy as np
import os
import pickle
#from yolov5.detect import detect
from process_detects_yolov8 import get_detected_img
from scipy.special import rel_entr, kl_div

from pathlib import Path
import argparse
import random
torch.set_grad_enabled(False)


threshold = 0.7
conf_dt =0.5 #confidence of detection
cnt_img =0

#-------------------------- config ------------------------------------------------
kld_thres = 34
num_features = 800

#---------------------------------------------------------------------------------


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


    '''cv2.imwrite('tmp/'+str(ttt)+'.jpg', dt_img0)
    cv2.imwrite('tmp/'+str(ttt)+'_1.jpg', dt_img1)'''

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

        #can not detect any matching keypoints between 2 images, or number of keypoints smaller than 2 -> different or not definitely the same
        if ((len(tmp_x1)<=2) or (len(tmp_x2)<=2)):
            return 1

        kld = sum(abs(rel_entr(tmp_x1, tmp_x2)))/len(tmp_x1)

        # Draw
        color = cm.jet(mconf)
        text = [
            'LoFTR',
            'Matches: {}'.format(len(mkpts0)),
        ]

        name_=""
        if kld > kld_thres:
            tmp = "anh"+path1.split('/')[-1].split('.')[0] +'_'+"anh"+path2.split('/')[-1].split('.')[0]
            name_ +="khongdat_" +str(str(kld).split('.')[0])+"diff_"+tmp  #da thay doi
        else:
            tmp = "anh"+path1.split('/')[-1].split('.')[0] +'_'+"anh"+path2.split('/')[-1].split('.')[0]
            name_ +="dat_" +str(str(kld).split('.')[0])+"diff_"+tmp  #da thay doi

        global cnt_img
        cnt_img +=1
        fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, text=text, 
                                   path='/home/nts1/users/datts/image_matching/LoFTR/res_img/'+name_+'.jpg')
        
        print("kld: ", kld)
         
        if kld > kld_thres:
            return 1 #da thay doi
        else:
            return 0 #khong thay doi



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--input_pairs', type=str, default='assets/scannet_sample_pairs_with_gt.txt',
        help='Path to the list of image pairs')
    parser.add_argument(
        '--input_dir', type=str, default='assets/scannet_sample_images/',
        help='Path to the directory that contains the images')
    parser.add_argument(
        '--output_dir', type=str, default='dump_match_pairs/',
        help='Path to the directory in which the .npz results and optionally,'
             'the visualization images are written')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')

    opt = parser.parse_args()
    print(opt)


    with open(opt.input_pairs, 'r') as f:
        pairs = [l.split() for l in f.readlines()]


    # Load the SuperPoint and LoFTR models.
    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))

    matcher = LoFTR(config=default_cfg)
    matcher.load_state_dict(torch.load("weights/outdoor_ds.ckpt")['state_dict'])
    matcher = matcher.eval().cuda()

    # run inference
    for idx, pair in enumerate(pairs):
        path1, path2 = os.path.join(opt.input_dir, pair[0]), os.path.join(opt.input_dir, pair[1])

        #---------------return result---------------------------
        res = inference(path1, path2, pair[0].split('.')[0]+'_'+pair[1].split('.')[0], conf_dt)
        print(path1+' '+ path2)
        print(res)

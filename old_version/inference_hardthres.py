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
from yolov5.detect import detect
from process_detects import get_detected_img
from scipy.special import rel_entr, kl_div

threshold = 0.7
conf_dt =0.5 #confidence of detection


data_path = 'test_img/img9'
pair_path = 'pairs/img9.txt'




num_features = 800
minirocket_clf = MiniRocketClassifier()

minirocket_clf=load_minirocket('800fe_87all')


xgb = pickle.load(open('xgb_perspective_yolov5_84acc_88pre_79rec.pkl', "rb"))




# Initialize LoFTR
matcher = LoFTR(config=default_cfg)
#matcher.load_state_dict(torch.load("weights/indoor_ds_new.ckpt")['state_dict'])
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

    


def inference(path1, path2, name, conf_dt):
    # Load example images
    img0_pth = path1
    img1_pth = path2



    dt_img0, x11, y11, x12, y12 = get_detected_img(img0_pth, conf_dt)

    #dt_img0 = cv2.imread(img0_pth)
    
    img0_raw = cv2.cvtColor(dt_img0, cv2.COLOR_BGR2GRAY)
    img0_shape = dt_img0.shape  #(H, W, C)

    '''if img1_pth == 'test_img/img8/2_1.jpg':
        print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
        print(img0_shape)
        print(dt_img0.shape)
        cv2.imwrite('./zzzzzzzzz.jpg', img0_raw)'''
    # print(">>>>>>>>>>>>>>>>.")
    # print(img0_shape)

    dt_img1, x21, y21, x22, y22 = get_detected_img(img1_pth, conf_dt)
    #dt_img1 = cv2.imread(img1_pth)
    img1_raw = cv2.cvtColor(dt_img1, cv2.COLOR_BGR2GRAY)
    img1_shape = dt_img1.shape

    

    img0_raw = cv2.resize(img0_raw, (640, 480))
    img1_raw = cv2.resize(img1_raw, (640, 480))

    # print(";::::::::::::::::::::::::::::")
    # print(img0_raw.shape)

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

        # tmp_x1[0::2] = tmp_x1[0::2]/640
        # tmp_x1[1::2] = tmp_x1[0::2]/480

        
        tmp_x1=tmp_x1[:num_features*2]
        tmp_x2=tmp_x2[:num_features*2]
        '''arr_row, num =  np.concatenate((tmp_x1, tmp_x2)), len(mconf)
        mean_x = arr_row.mean()
        padd_arr = np.zeros((num_features*4))
        padd_arr.fill(mean_x)
        if num<=num_features:
            padd_arr[:num*4] = arr_row
        else:
            padd_arr[:] = arr_row[:num_features*4]'''
        #---------------------------------------------------------------------------------------
        # print(num)
        '''ex1, ey1, stdx1, stdy1 = tmp_x1[0::2].mean(), tmp_x1[1::2].mean(), np.std(tmp_x1[0::2]), np.std(tmp_x1[1::2])
        ex2, ey2, stdx2, stdy2 = tmp_x2[0::2].mean(), tmp_x2[1::2].mean(), np.std(tmp_x2[0::2]), np.std(tmp_x2[1::2])'''

        #----------------------------------------------------------------------- add the difference compared to identity matrix-------------------------------------------------------------------------------
        avg = 0
        num_matrix_to_check = 50
        num_greater_1000 =0
        list_dis =[]
        
        for ii_ in range(num_matrix_to_check):
            rand_arr = np.random.choice(range(len(tmp_x1)//2), 4, replace=False)
            pts1 = np.array([
                [tmp_x1[2*rand_arr[0]], tmp_x1[2*rand_arr[0]+1] ],
                [tmp_x1[2*rand_arr[1]], tmp_x1[2*rand_arr[1]+1] ],
                [tmp_x1[2*rand_arr[2]], tmp_x1[2*rand_arr[2]+1] ],
                [tmp_x1[2*rand_arr[3]], tmp_x1[2*rand_arr[3]+1] ],

                
                ]).astype(np.float32)
            pts2 = np.array([
                [tmp_x2[2*rand_arr[0]], tmp_x2[2*rand_arr[0]+1] ],
                [tmp_x2[2*rand_arr[1]], tmp_x2[2*rand_arr[1]+1] ],
                [tmp_x2[2*rand_arr[2]], tmp_x2[2*rand_arr[2]+1] ],
                [tmp_x2[2*rand_arr[3]], tmp_x2[2*rand_arr[3]+1] ],

                
                ]).astype(np.float32)
            
            # print("///////////////////////")
            # print(pts1)
            # print(pts1.shape)
            
            dis = calculate_perspective_dis(pts1, pts2)
            #print(dis)
            list_dis.append(dis)
            if dis>1000:
                num_greater_1000 +=1
            avg +=dis
        avg = avg/num_matrix_to_check
        #print("final avg: ", avg)

        list_dis=np.array(list_dis)
        num_gr=sum(list_dis>avg)
        num_sm = sum(list_dis<=avg)
        print("num greater avg: ", num_gr)
        print("num smaller avg: ", num_sm)

        weighted = sum(list_dis[list_dis>avg])*num_gr/len(list_dis) + sum(list_dis[list_dis<=avg])*num_sm/len(list_dis)
        print("weighted: ", weighted)
        print("mean weighted: ", weighted/(len(list_dis)*avg))
        print("KLD: ", sum(abs(rel_entr(tmp_x1, tmp_x2)))/len(tmp_x1))
        #print("KLD 1: ", sum(kl_div(tmp_x1, tmp_x2)))
        
        
        #padd_arr = np.array([[ex1, ey1, stdx1, stdy1, ex2, ey2, stdx2, stdy2, 0]])




        # pred_mini = minirocket_clf.predict(padd_arr)
        # #pred_mini = xgb.predict(padd_arr)

        # print(path1+' '+path2)
        # print(pred_mini)
        #-------------------------------------------------------------------------------------

        # Draw
        color = cm.jet(mconf)
        text = [
            'LoFTR',
            'Matches: {}'.format(len(mkpts0)),
        ]

        '''if img1_pth == 'test_img/img8/2_1.jpg':
            print(img0_raw.shape)
            cv2.imwrite('./xxxxxxxx.jpg', img0_raw)
            cv2.imwrite('./yyyyyyyy.jpg', img1_raw)'''
        fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, text=text, 
                                   path='/home/nts1/users/datts/image_matching/LoFTR/res_img/'+name+'_pred'+str(str(avg).split('.')[0]) +'.jpg')
        
    return fig


f=open(pair_path, "r")
lines=f.readlines()
f.close()

for idx, line in enumerate(lines):
    imgs = line[:-1].split(' ')
    path1, path2 = os.path.join(data_path, imgs[0]), os.path.join(data_path, imgs[1])
    print("------------------------------------------------------------------")
    print(path1, path2)
    inference(path1, path2, imgs[0].split('.')[0]+'_'+imgs[1].split('.')[0], conf_dt)
    
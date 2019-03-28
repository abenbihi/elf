
import os
import argparse
import time
import random
import cv2
import numpy as np

from utils.dump import saveh5, loadh5

import tools.cst as cst 
import tools.bench_tools as bench_tools

parser = argparse.ArgumentParser()
parser.add_argument('--lift_data_id', type=str, default='0', help='id of lift output directory')
parser.add_argument('--trials', type=str, default='100', help='xp id')
parser.add_argument('--max_num_feat', type=int, default=1000, help='maximum number of features to keep')
parser.add_argument('--h', type=int, default=480, help='new height')
parser.add_argument('--w', type=int, default=704, help='new width')
parser.add_argument('--resize', type=int, default=0, help='set to 1 to resize img')
args = parser.parse_args()

new_size = (args.w,args.h)

# setup output dir
res_dir = 'res/lift/%s'%args.trials
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

# write human readable logs
f = open(os.path.join(res_dir, 'log.txt'), 'w')
f.write('method: lift\n')
f.write('data: %s\n'%cst.DATA)
f.write('thresh_overlap: %d\n'%cst.THRESH_OVERLAP)
f.write('thresh_desc: %d\n'%cst.THRESH_DESC)


norm = 'L2'
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
matcher = cv2.FlannBasedMatcher(index_params,search_params)


i_des_dir = os.path.join('res/lift/', args.lift_data_id, 'des_no_aug')
v_des_dir = os.path.join('res/lift/', args.lift_data_id, 'des_aug')


global_start_time = time.time()
for scene_name in cst.SCENE_LIST:
    duration = time.time() - global_start_time
    minutes = duration / 60
    seconds = duration % 60
    print('************ %s ************ %d:%02d'%(scene_name,
        minutes, seconds))
    f.write('************ %s ************\n'%scene_name)


    if cst.DATA=='hpatches':
        if scene_name[0]=='i':
            lift_dir = i_des_dir
        elif scene_name[0]=='v':
            lift_dir = v_des_dir
        else:
            print('Error: There is something wrong with the scene name: %s'%scene_name)
            exit(1)
    elif cst.DATA=='hpatches_rot' or cst.DATA=='hpatches_s':
        lift_dir = v_des_dir

        
    img0_fn = os.path.join(cst.DATA_DIR, scene_name,'%d.ppm'%1)
    print('img_fn: %s'%img0_fn)
    img0 = cv2.imread(img0_fn)
    old_size0 = (img0.shape[1], img0.shape[0])
    if args.resize==1:
        img0 = cv2.resize(img0, new_size, interpolation=cv2.INTER_LINEAR)
    lift_out0_fn = os.path.join(lift_dir, scene_name, '%d.h5'%1)
    lift_out0 = loadh5(lift_out0_fn)

    # detection and description
    kp0 = []
    for line in lift_out0['keypoints'][:args.max_num_feat,:]:
        kp = cv2.KeyPoint(x=line[0],y=line[1], _size=2,
                _angle=0, _response=0, _octave=0, _class_id=0)
        kp0.append(kp)
    des0 = lift_out0['descriptors'][:args.max_num_feat,:]

    # draw kp on img
    kp_on_img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    kp_on_img0 = np.tile(np.expand_dims(kp_on_img0,2), (1,1,3))
    for i,kp in enumerate(kp0):
        pt = (int(round(kp.pt[0])), int(round(kp.pt[1])))
        cv2.circle(kp_on_img0, pt, 1, (0, 255, 0), -1, lineType=16)

    # get data
    for img_key in range(2,cst.MAX_IMG_NUM+1):
        if cst.DATA=='hpatches': 
            # there is a bug with these scenes, I don't know why
            if ((scene_name =='v_feast' and img_key==6) or
                    ((scene_name =='v_wall' and img_key==6))):
                continue
        H = np.loadtxt(os.path.join(cst.DATA_DIR, scene_name, 'H_1_%d'%img_key))
        
        img1_fn = os.path.join(cst.DATA_DIR, scene_name,'%d.ppm'%img_key)
        img1 = cv2.imread(img1_fn)
        lift_out1_fn = os.path.join(lift_dir, scene_name, '%d.h5'%img_key)
        lift_out1 = loadh5(lift_out1_fn)

        # resize it
        if args.resize==1:
            s1x = 1.0*new_size[0]/old_size0[0]
            s1y = 1.0*new_size[1]/old_size0[1]
            six = 1.0*new_size[0]/img1.shape[1]
            siy = 1.0*new_size[1]/img1.shape[0]
            #print('s1x - s1y - six - siy', s1x, s1y, six, siy)
            H = np.diag((six,siy,1)).dot(H.dot( np.diag((1.0/s1x, 1.0/s1y, 1)) ) )
            img1 = cv2.resize(img1, new_size, interpolation=cv2.INTER_LINEAR)
       

        # detection
        kp1 = []
        for line in lift_out1['keypoints'][:args.max_num_feat,:]:
            kp = cv2.KeyPoint(x=line[0],y=line[1], _size=2,
                    _angle=0, _response=0, _octave=0, _class_id=0)
            kp1.append(kp)
        des1 = lift_out1['descriptors'][:args.max_num_feat,:]

        # draw kp on img
        kp_on_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        kp_on_img1 = np.tile(np.expand_dims(kp_on_img1,2), (1,1,3))
        for i,kp in enumerate(kp1):
            pt = (int(round(kp.pt[0])), int(round(kp.pt[1])))
            cv2.circle(kp_on_img1, pt, 1, (0, 255, 0), -1, lineType=16)
        
        # metrics
        print('** %d **'%img_key)
        f.write('** %d **\n'%img_key)

        rep, N1, N2, M = bench_tools.rep(new_size, H,
                kp0, kp1, cst.THRESH_OVERLAP)
        print('rep: %.3f - N1: %d - N2: %d - M: %d'%(rep,N1,N2,len(M)))
        f.write('rep:%.3f - N1:%d - N2:%d - M:%d\n' %(rep,N1,N2,len(M)))


        (ms, N1, N2, M_len, M_d_len, inter) = bench_tools.ms(new_size, H,
                kp0, kp1, 
                des0, des1,
                cst.THRESH_OVERLAP, cst.THRESH_DESC)
        print('ms:%.3f - N1:%d - N2:%d - M:%d - M_d:%d - inter:%d'
                %(ms,N1,N2,M_len, M_d_len, inter))
        f.write('ms:%.3f - N1:%d - N2:%d - M:%d - M_d:%d - inter:%d\n'
            %(ms, N1, N2, M_len, M_d_len, inter))


        if cst.DEBUG:
            good = []
            matches = matcher.knnMatch(des0, des1,k=2)
            for i,(m,n) in enumerate(matches):
                if m.distance < 0.8*n.distance:
                    good.append(m)

            match_des_img = cv2.drawMatches(img0, kp0,
                    img1, kp1,
                    good, None, flags=2)
            cv2.imshow('match_des', match_des_img)
            cv2.imshow('kp_on', np.hstack((kp_on_img0,
                kp_on_img1)))
            cv2.waitKey(0)


f.close()
            



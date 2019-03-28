
import os
import argparse
import time
import cv2
import numpy as np

from utils.dump import saveh5, loadh5

import tools.cst as cst 
import tools.bench_tools_no_H as bench_tools

parser = argparse.ArgumentParser()
parser.add_argument('--lift_data_id', type=str, default='0', help='id of lift output directory')
parser.add_argument('--trials', type=str, default='100', help='xp id')
parser.add_argument('--max_num_feat', type=int, default=1000, help='maximum number of features to keep')
parser.add_argument('--h', type=int, default=480, help='new height')
parser.add_argument('--w', type=int, default=704, help='new width')
parser.add_argument('--resize', type=int, default=704, help='Set to 1 to resize.')
args = parser.parse_args()

new_size = (args.w, args.h)

# get data, setup output dir
res_dir = 'res/lift/%s'%args.trials
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

# write human readable logs
f = open(os.path.join(res_dir, 'log.txt'), 'w')
f.write('lift\n')
f.write('thresh_overlap: %d\n'%cst.THRESH_OVERLAP)
f.write('thresh_desc: %d\n'%cst.THRESH_DESC)


norm = 'L2'
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
matcher = cv2.FlannBasedMatcher(index_params,search_params)


i_des_dir = os.path.join('res/lift', args.lift_data_id, 'des_no_aug')
v_des_dir = os.path.join('res/lift', args.lift_data_id, 'des_aug')
lift_dir = v_des_dir


global_start_time = time.time()
for scene_name in cst.SCENE_LIST:
    
    duration = time.time() - global_start_time
    minutes = duration / 60
    seconds = duration % 60
    print('************ %s ************ %d:%02d'%(scene_name,
        minutes, seconds))
    f.write('************ %s ************\n'%scene_name)

    scene_dir = '%s/%s/'%(cst.DATA_DIR, scene_name)
    pose_fn = '%s/images.txt'%(scene_dir)
    img_list = [ l for l in sorted(os.listdir(scene_dir)) if l[-3:]=='png']
    img_num = len(img_list)

    # intrinsic params (same for all img)
    camera_fn   = os.path.join(scene_dir, '0000.png.camera')
    #print('camera_fn: %s'%camera_fn)
    K, T = bench_tools.load_camera(camera_fn)
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]   
    K_inv = np.linalg.inv(K)

    img0_root_fn = '%04d'%0
    img0_fn = os.path.join(scene_dir, '%s.png'%img0_root_fn)
    img0 = cv2.imread(img0_fn)
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    img0_big = img0.copy()
    old_size = (img0.shape[1], img0.shape[0])
    img0 = cv2.resize(img0, new_size, interpolation=cv2.INTER_AREA)
    
    sx2big, sy2big = 1.0*old_size[0]/new_size[0], 1.0*old_size[1]/new_size[1] 
    sx2small, sy2small = 1.0*new_size[0]/old_size[0], 1.0*new_size[1]/old_size[1]
    print('sx2big: %.3f - sy2big: %.3f'%(sx2big, sy2big))
    print('sx2small: %.3f - sy2small: %.3f'%(sx2small, sy2small))
    #raw_input('wait')
    
    # get colmap depth map
    depth0_fn    = '%s/%s.png.photometric.bin'%(scene_dir, img0_root_fn)
    depth0_map = bench_tools.read_array(depth0_fn)
    depth0_map = cv2.resize(depth0_map, old_size, interpolation=cv2.INTER_CUBIC)
    print('depth0_map.shape', depth0_map.shape)
    #raw_input('wait')
    
    # Get camera pose
    kp_all_l = [l.split("\n")[0] for l in open(pose_fn).readlines()][4:]
    header0, kp0_l = [],[]
    for i, l in enumerate(kp_all_l):
        if l.split(" ")[-1] == ('%s.png'%img0_root_fn):
            header0 = l
            kp0_l = kp_all_l[i+1]
    kp0_l = kp0_l.split(" ")
    kp0_l = np.reshape(np.array(kp0_l), [-1,3]).astype(float)
    q0 = np.array(header0.split(" ")[1:5]).astype(np.float)
    t0 = np.array(header0.split(" ")[5:8]).astype(np.float)
    R0 = bench_tools.qvec2rotmat(q0)


    # detect and describe
    lift_out0_fn = os.path.join(lift_dir, scene_name, '%04d.h5'%0)
    lift_out0 = loadh5(lift_out0_fn)
    kp0 = []
    for line in lift_out0['keypoints'][:args.max_num_feat,:]:
        kp = cv2.KeyPoint(x=line[0],y=line[1], _size=2,
                _angle=0, _response=0, _octave=0, _class_id=0)
        kp0.append(kp)
    des0 = lift_out0['descriptors'][:args.max_num_feat,:]


    # draw kp
    kp_on_img0 = np.tile(np.expand_dims(img0,2), (1,1,3))
    for i,kp in enumerate(kp0):
        pt = (int(round(kp.pt[0])), int(round(kp.pt[1])))
        cv2.circle(kp_on_img0, pt, 1, (0, 255, 0), -1, lineType=16)


    for img_id in range(1,img_num):
        print('** %04d ** '%img_id)
        f.write('** %04d **\n'%img_id)

        # get img
        img1_root_fn = '%04d'%img_id
        img1_fn = os.path.join(scene_dir, '%s.png'%img1_root_fn)
        img1 = cv2.imread(img1_fn)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img1_big = img1.copy()
        img1 = cv2.resize(img1, new_size, interpolation=cv2.INTER_AREA)
        
        # get depth
        depth1_fn    = '%s/%s.png.photometric.bin'%(scene_dir, img0_root_fn)
        depth1_map = bench_tools.read_array(depth1_fn)
        depth1_map = cv2.resize(depth1_map, old_size, interpolation=cv2.INTER_CUBIC)
        #print('depth1_map.shape', depth1_map.shape)
        #raw_input('wait')
       
        # get camera pose
        header1, kp1_l = [],[] # camera pose, pixels list
        for i, l in enumerate(kp_all_l):
            if l.split(" ")[-1] == ('%s.png'%img1_root_fn):
                header1 = l
                kp1_l = kp_all_l[i+1]
        kp1_l = kp1_l.split(" ")
        kp1_l = np.reshape(np.array(kp1_l), [-1,3]).astype(float)
        q1 = np.array(header1.split(" ")[1:5]).astype(np.float)
        t1 = np.array(header1.split(" ")[5:8]).astype(np.float)
        R1 = bench_tools.qvec2rotmat(q1)
   

        # detect and describe
        lift_out1_fn = os.path.join(lift_dir, scene_name, '%04d.h5'%img_id)
        lift_out1 = loadh5(lift_out1_fn)
        kp1 = []
        for line in lift_out1['keypoints'][:args.max_num_feat,:]:
            kp = cv2.KeyPoint(x=line[0],y=line[1], _size=2,
                    _angle=0, _response=0, _octave=0, _class_id=0)
            kp1.append(kp)
        des1 = lift_out1['descriptors'][:args.max_num_feat,:]


        # draw kp
        kp_on_img1 = np.tile(np.expand_dims(img1,2), (1,1,3))
        for i,kp in enumerate(kp1):
            pt = (int(round(kp.pt[0])), int(round(kp.pt[1])))
            cv2.circle(kp_on_img1, pt, 1, (0, 255, 0), -1, lineType=16)
    

        # scale kp to original size (i.e. the big img) for projection
        # let's go
        h_pts0 = np.vstack([[sx2big*kp.pt[0], sy2big*kp.pt[1], 1] for kp in kp0])
        h_pts1 = np.vstack([[sx2big*kp.pt[0], sy2big*kp.pt[1], 1] for kp in kp1])
    

        rep, N1, N2, M = bench_tools.rep(old_size, h_pts0, h_pts1, K, depth0_map,
                depth1_map, R0, t0, R1, t1, cst.THRESH_OVERLAP*sx2big)
        print('rep: %.3f - N1: %d - N2: %d - M: %d'%(rep,N1,N2,M))
        f.write('rep:%.3f - N1:%d - N2:%d - M:%d\n' %(rep,N1,N2,M))
        
        ms, N1, N2, M_len, M_d_len, inter = bench_tools.ms(old_size, des0,
                des1, h_pts0, h_pts1, K, depth0_map, depth1_map, R0, t0, R1,
                t1, cst.THRESH_OVERLAP*sx2big, cst.THRESH_DESC, norm)
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

            match_des_img = cv2.drawMatches(img0, kp0, img1, kp1,
                    good, None, flags=2)
            cv2.imshow('match_des', match_des_img)
            cv2.imshow('kp_on_img', np.hstack((kp_on_img0,
                kp_on_img1)))
            cv2.waitKey(0)


f.close()
            



"""
Compute ms on the run of superpoint
kp: superpoint
des: superpoint
"""
import argparse
import glob
import numpy as np
import os
import time

import cv2
import torch

import tools.bench_tools_no_H as bench_tools
import tools.cst as cst
import methods.superpoint.tools as tools

# Parse command line arguments.
parser = argparse.ArgumentParser(description='PyTorch SuperPoint Demo.')
parser.add_argument('--weights_path', type=str, default='superpoint_v1.pth',
    help='Path to pretrained weights file (default: superpoint_v1.pth).')
parser.add_argument('--H', type=int, default=120, help='Input image height (default: 120).')
parser.add_argument('--W', type=int, default=160, help='Input image width (default:160).')
parser.add_argument('--nms_dist', type=int, default=4,
    help='Non Maximum Suppression (NMS) distance (default: 4).')
parser.add_argument('--border_remove', type=int, default=4, help='Border of the img where you ignore the kp')
parser.add_argument('--max_num_features', type=int, default=1000, help='Maximum number of features.')
parser.add_argument('--conf_thresh', type=float, default=0.015, help='Detector confidence threshold (default: 0.015).')
parser.add_argument('--nn_thresh', type=float, default=0.7, help='Descriptor matching threshold (default: 0.7).')
parser.add_argument('--trials', type=str, default=0)
parser.add_argument('--resize', type=int, default=0, help='set to 1 to resize img')
opt = parser.parse_args()
print(opt)

new_size = (opt.W, opt.H)
norm = 'L2'
 
# setup output dir
res_dir = os.path.join('res/superpoint/', opt.trials)
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

# write human readable logs
f = open(os.path.join(res_dir, 'log.txt'), 'w')
f.write('superpoint\n')
f.write('thresh_overlap: %d\n'%cst.THRESH_OVERLAP)
f.write('thresh_desc: %d\n'%cst.THRESH_DESC)


# matcher
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
matcher = cv2.FlannBasedMatcher(index_params,search_params)


# This class helps load input images from different sources.
vs = tools.VideoStreamer('', 0, opt.H, opt.W, 1, '*ppm')
print('==> Loading pre-trained network.')
# This class runs the SuperPoint network and processes its outputs.
fe = tools.SuperPointFrontend(weights_path=opt.weights_path,
                        nms_dist=opt.nms_dist,
                        conf_thresh=opt.conf_thresh,
                        nn_thresh=opt.nn_thresh,
                        cuda=True)
print('==> Successfully loaded pre-trained network.')
print('==> Running Demo.')


global_start_time = time.time()
for scene_name in cst.SCENE_LIST:

    print('************ %s ************'%scene_name)
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


    #get 1st img
    img0_root_fn = '%04d'%0
    img0_fn = os.path.join(scene_dir, '%s.png'%img0_root_fn)
    img0 = cv2.imread(img0_fn)
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    img0_big = img0.copy()
    old_size = (img0.shape[1], img0.shape[0])
    img0 = cv2.resize(img0, new_size, interpolation=cv2.INTER_AREA)
    
    # resize ratios
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
    #kp0, des0 = sift.detectAndCompute(img0,None)
    img = vs.read_image(img0_fn, (opt.H, opt.W))
    pts0, des0, heatmap0 = fe.run(img)
    pts0 = pts0[:,:opt.max_num_features]
    des0 = des0[:,:opt.max_num_features].T
    # convert to cv2 kp for prototype conformity
    kp0 = []
    for pt in pts0.T:
        kp = cv2.KeyPoint(x=pt[0],y=pt[1], _size=2,
            _angle=0, _response=0, _octave=0, _class_id=0)
        kp0.append(kp)

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
        img = vs.read_image(img1_fn, (opt.H, opt.W))
        pts1, des1, heatmap1 = fe.run(img)
        pts1 = pts1[:,:opt.max_num_features]
        des1 = des1[:,:opt.max_num_features].T
        # convert to cv2 kp for prototype conformity
        kp1 = []
        for pt in pts1.T:
            kp = cv2.KeyPoint(x=pt[0],y=pt[1], _size=2,
                _angle=0, _response=0, _octave=0, _class_id=0)
            kp1.append(kp)
   
        if len(kp0)==0 or len(kp1)==0:
            N1 = len(kp0)
            N2 = len(kp1)
            ms = 0
            M, M_d, inter = [],[],[]
            continue
    
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
                t1, cst.THRESH_OVERLAP*sx2big, cst.THRESH_DESC, norm='L2')
        print('ms:%.3f - N1:%d - N2:%d - M:%d - M_d:%d - inter:%d'
                %(ms,N1,N2,M_len, M_d_len, inter))
        f.write('ms:%.3f - N1:%d - N2:%d - M:%d - M_d:%d - inter:%d\n'
            %(ms, N1, N2, M_len, M_d_len, inter))

        if cst.DEBUG:
            # match sift
            good = []
            matches = matcher.knnMatch(des0, des1,k=2)
            for i,(m,n) in enumerate(matches):
                if m.distance < 0.8*n.distance:
                    good.append(m)
            match_des_img = cv2.drawMatches(img0, kp0, img1, kp1,
                    good, None, flags=2)
            cv2.imshow('match_des', match_des_img)
            cv2.imshow('kp_on_img0', np.hstack((kp_on_img0, kp_on_img1)))
            cv2.imshow('heatmap', np.hstack((heatmap0, heatmap1)))
            cv2.waitKey(0)

print('==> Finshed Demo.')
f.close()

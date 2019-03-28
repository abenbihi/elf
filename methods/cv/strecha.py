
import os
import argparse
import time
import cv2
import numpy as np

import tools.bench_tools_no_H as bench_tools
import tools.cst as cst

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, help='opencv detector/descriptor')
parser.add_argument('--trials', type=str, default='100', help='xp id')
parser.add_argument('--max_num_feat', type=int, default=1000, help='maximum number of features to keep')
parser.add_argument('--h', type=int, default=480, help='new height')
parser.add_argument('--w', type=int, default=704, help='new width')
parser.add_argument('--resize', type=int, default=0, help='set to 1 to resize img')
args = parser.parse_args()

method = args.method
new_size = (args.w, args.h)

# setup output dir
res_dir = 'res/%s/%s'%(method, args.trials)
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

# write human readable logs
f = open(os.path.join(res_dir, 'log.txt'), 'w')
f.write('method: %s\n'%method)
f.write('data: %s\n'%cst.DATA)
f.write('thresh_overlap: %d\n'%cst.THRESH_OVERLAP)
f.write('thresh_desc: %d\n'%cst.THRESH_DESC)


# feature extraction handler
if method=='sift':
    fe = cv2.xfeatures2d.SIFT_create(args.max_num_feat)
elif method=='surf':
    fe = cv2.xfeatures2d.SURF_create(400)
elif method=='orb':
    fe = cv2.ORB_create()
elif method=='mser':
    fe = cv2.MSER_create()
elif method=='akaze':
    fe = cv2.AKAZE_create()
else:
    print('This mtf method is not handled: %s'%method)
    exit(1)

# feature matcher handler for visualization
if method == 'orb':
    norm = 'hamming'
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
elif method=='akaze':
    norm = 'hamming'
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
else:
    norm = 'L2'
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params,search_params)


global_start_time = time.time()
for scene_name in cst.SCENE_LIST:
    duration = time.time() - global_start_time
    minutes = duration / 60
    seconds = duration % 60
    print('*** %s *** %d:%02d'%(scene_name, duration/60, duration%60))
    f.write('*** %s *** %d:%02d\n'%(scene_name, duration/60, duration%60))

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
    print(old_size)
   
    # resize ratios
    sx2big, sy2big = 1.0*old_size[0]/new_size[0], 1.0*old_size[1]/new_size[1] 
    sx2small, sy2small = 1.0*new_size[0]/old_size[0], 1.0*new_size[1]/old_size[1]
    print('sx2big: %.3f - sy2big: %.3f'%(sx2big, sy2big))
    print('sx2small: %.3f - sy2small: %.3f'%(sx2small, sy2small))
    
    # get colmap depth map, resize it to original img size
    depth0_fn    = '%s/%s.png.photometric.bin'%(scene_dir, img0_root_fn)
    depth0_map = bench_tools.read_array(depth0_fn)
    depth0_map = cv2.resize(depth0_map, old_size, interpolation=cv2.INTER_CUBIC)
    print('depth0_map.shape', depth0_map.shape)
    
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
    kp0, des0 = fe.detectAndCompute(img0,None)
    kp0 = kp0[:args.max_num_feat]
    des0 = des0[:args.max_num_feat, :]

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
        kp1, des1 = fe.detectAndCompute(img1,None)
        kp1 = kp1[:args.max_num_feat]
        des1 = des1[:args.max_num_feat, :]

        # draw kp
        kp_on_img1 = np.tile(np.expand_dims(img1,2), (1,1,3))
        for i,kp in enumerate(kp1):
            pt = (int(round(kp.pt[0])), int(round(kp.pt[1])))
            cv2.circle(kp_on_img1, pt, 1, (0, 255, 0), -1, lineType=16)

    
        # scale kp to original size (i.e. the big img) for projection
        h_pts0 = np.vstack([[sx2big*kp.pt[0], sy2big*kp.pt[1], 1] for kp in kp0])
        h_pts1 = np.vstack([[sx2big*kp.pt[0], sy2big*kp.pt[1], 1] for kp in kp1])
    
    
        print('** %d **'%img_id)
        f.write('** %d **\n'%img_id)

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
            if method=='orb':
                matches = matcher.match(des0,des1)
                # Sort them in the order of their distance.
                # these are not necessarily good matches, I just called them
                # good to be homogeneous
                good = sorted(matches, key = lambda x:x.distance)
            else:
                matches = matcher.knnMatch(des0, des1,k=2)
                for i,(m,n) in enumerate(matches):
                    if m.distance < 0.8*n.distance:
                        good.append(m)

            match_des_img = cv2.drawMatches(img0, kp0, img1, kp1,
                    good, None, flags=2)
            cv2.imshow('match_des', match_des_img)
            cv2.imshow('kp_on_img0', np.hstack((kp_on_img0,
                kp_on_img1)))
            cv2.waitKey(0)


f.close()
            



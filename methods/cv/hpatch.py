
import os
import argparse
import time
import cv2
import numpy as np

import tools.bench_tools as bench_tools
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

    # get 1st img, (resize it), convert to BW
    img0_fn = os.path.join(cst.DATA_DIR, scene_name,'%d.ppm'%1)
    print('img_fn: %s'%img0_fn)
    img0 = cv2.imread(img0_fn)
    old_size0 = (img0.shape[1], img0.shape[0])
    if args.resize==1:
        img0 = cv2.resize(img0, new_size, interpolation=cv2.INTER_LINEAR)
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)


    # detect and describe
    kp0, des0 = fe.detectAndCompute(img0,None)
    kp0 = kp0[:args.max_num_feat]
    des0 = des0[:args.max_num_feat, :]
    kp_on_img0 = np.tile(np.expand_dims(img0,2), (1,1,3))
    # draw kp on img
    for i,kp in enumerate(kp0):
        pt = (int(round(kp.pt[0])), int(round(kp.pt[1])))
        cv2.circle(kp_on_img0, pt, 1, (0, 255, 0), -1, lineType=16)


    for img_key in range(2,cst.MAX_IMG_NUM+1):
        # get 2nd img, (resize it), convert to BW
        img1_fn = os.path.join(cst.DATA_DIR, scene_name,'%d.ppm'%img_key)
        #print('img_fn: %s'%img1_fn)
        img1 = cv2.imread(img1_fn)
        H = np.loadtxt(os.path.join(cst.DATA_DIR, scene_name, 'H_1_%d'%img_key))

        if args.resize==1:
            s1x = 1.0*new_size[0]/old_size0[0]
            s1y = 1.0*new_size[1]/old_size0[1]
            six = 1.0*new_size[0]/img1.shape[1]
            siy = 1.0*new_size[1]/img1.shape[0]
            #print('s1x - s1y - six - siy', s1x, s1y, six, siy)
            H = np.diag((six,siy,1)).dot(H.dot( np.diag((1.0/s1x, 1.0/s1y, 1)) ) )
            img1 = cv2.resize(img1, new_size, interpolation=cv2.INTER_LINEAR)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)


        # detect and describe
        kp1, des1 = fe.detectAndCompute(img1,None)
        kp1 = kp1[:args.max_num_feat]
        des1 = des1[:args.max_num_feat, :]


        # draw kp
        kp_on_img1 = np.tile(np.expand_dims(img1,2), (1,1,3))
        for i,kp in enumerate(kp1):
            pt = (int(round(kp.pt[0])), int(round(kp.pt[1])))
            cv2.circle(kp_on_img1, pt, 1, (0, 255, 0), -1, lineType=16)


        # metrics
        print('** %d **'%img_key)
        f.write('** %d **\n'%img_key)

        rep, N1, N2, M = bench_tools.rep(new_size, H, kp0, kp1, 
                cst.THRESH_OVERLAP)
        print('rep: %.3f - N1: %d - N2: %d - M: %d'%(rep,N1,N2,len(M)))
        f.write('rep:%.3f - N1:%d - N2:%d - M:%d\n' %(rep,N1,N2,len(M)))

        (ms, N1, N2, M_len, M_d_len, inter) = bench_tools.ms(new_size, H, 
                kp0, kp1, des0, des1, cst.THRESH_OVERLAP, cst.THRESH_DESC, norm)
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

            match_des_img = cv2.drawMatches(img0, kp0, img1, kp1, good, None, 
                    flags=2)

            cv2.imshow('match_des', match_des_img)
            cv2.imshow('kp_on', np.hstack((kp_on_img0, kp_on_img1)))
            cv2.waitKey(0)

f.close()
            



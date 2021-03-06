"""
Compute ms, rep and show me stuff
kp: elf (from file)
des: superpoint
"""
import argparse
import glob
import numpy as np
import os
import time

import cv2
import torch

import tools.bench_tools as bench_tools
import tools.cst as cst
import methods.superpoint.tools as tools


if __name__ == '__main__':
    
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description='PyTorch SuperPoint Demo.')
    parser.add_argument('--weights_path', type=str, default='superpoint_v1.pth',
        help='Path to pretrained weights file (default: superpoint_v1.pth).')
    parser.add_argument('--H', type=int, default=120,
        help='Input image height (default: 120).')
    parser.add_argument('--W', type=int, default=160,
        help='Input image width (default:160).')
    parser.add_argument('--nms_dist', type=int, default=4,
        help='Non Maximum Suppression (NMS) distance (default: 4).')
    parser.add_argument('--border_remove', type=int, default=4,
        help='Border of the img where you ignore the kp')
    parser.add_argument('--max_num_features', type=int, default=1000,
        help='Maximum number of features.')
    parser.add_argument('--conf_thresh', type=float, default=0.015,
        help='Detector confidence threshold (default: 0.015).')
    parser.add_argument('--nn_thresh', type=float, default=0.7,
        help='Descriptor matching threshold (default: 0.7).')
    parser.add_argument('--trials', type=str, default=0)
    parser.add_argument('--kp_dir_id', type=str, 
            default='kp are stored in res/elf/kp_dir_id')
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

    kp_dir = 'res/elf/%s'%opt.kp_dir_id


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
        duration = time.time() - global_start_time
        print('*** %s *** %d:%02d'%(scene_name, duration/60, duration%60))
        f.write('*** %s *** %d:%02d\n'%(scene_name, duration/60, duration%60))

        # get 1st img, (resize it), convert to BW
        img0_fn = os.path.join(cst.DATA_DIR, scene_name,'%d.ppm'%1)
        #print('img_fn: %s'%img0_fn)
        img0 = cv2.imread(img0_fn)
        old_size0 = (img0.shape[1], img0.shape[0])
        if opt.resize==1:
            img0 = cv2.resize(img0, new_size, interpolation=cv2.INTER_LINEAR)

        
        # detect
        pts_fn = os.path.join(kp_dir, scene_name, '%d_kp.txt'%1)
        pts0 = np.loadtxt(pts_fn)

        # convert to cv2 kp for prototype conformity
        kp0 = []
        for line in pts0.T:
            kp = cv2.KeyPoint(x=line[0],y=line[1], _size=2,
                    _angle=0, _response=0, _octave=0, _class_id=0)
            kp0.append(kp)

        # draw kp on img
        kp_on_img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        kp_on_img0 = np.tile(np.expand_dims(kp_on_img0,2), (1,1,3))
        for i,kp in enumerate(kp0):
            pt = (int(round(kp.pt[0])), int(round(kp.pt[1])))
            cv2.circle(kp_on_img0, pt, 1, (0, 255, 0), -1, lineType=16)


        # description
        img = vs.read_image(img0_fn, (opt.H, opt.W))
        _, _, heatmap, des0 = fe.run(img, pts0)
        des0 = des0.T


        for img_key in range(2, cst.MAX_IMG_NUM+1):
            # get 2nd img, (resize it), convert to BW
            img1_fn = os.path.join(cst.DATA_DIR, scene_name,'%d.ppm'%img_key)
            #print('img_fn: %s'%img1_fn)
            img1 = cv2.imread(img1_fn)
            H = np.loadtxt(os.path.join(cst.DATA_DIR, scene_name, 'H_1_%d'%img_key))
            
            # rectify H
            if opt.resize==1:
                s1x = 1.0*new_size[0]/old_size0[0]
                s1y = 1.0*new_size[1]/old_size0[1]
                six = 1.0*new_size[0]/img1.shape[1]
                siy = 1.0*new_size[1]/img1.shape[0]
                #print('s1x - s1y - six - siy', s1x, s1y, six, siy)
                H = np.diag((six,siy,1)).dot(H.dot( np.diag((1.0/s1x, 1.0/s1y, 1)) ) )
                img1 = cv2.resize(img1, new_size, interpolation=cv2.INTER_LINEAR)

            # detect
            pts_fn = os.path.join(kp_dir, scene_name, '%d_kp.txt'%img_key)
            pts1 = np.loadtxt(pts_fn)

            # convert to cv2 kp for prototype conformity
            kp1 = []
            for line in pts1.T:
                kp = cv2.KeyPoint(x=line[0],y=line[1], _size=2,
                        _angle=0, _response=0, _octave=0, _class_id=0)
                kp1.append(kp)


            # draw kp on img
            kp_on_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            kp_on_img1 = np.tile(np.expand_dims(kp_on_img1,2), (1,1,3))
            for i,kp in enumerate(kp1):
                pt = (int(round(kp.pt[0])), int(round(kp.pt[1])))
                cv2.circle(kp_on_img1, pt, 1, (0, 255, 0), -1, lineType=16)

            
            # description
            img = vs.read_image(img1_fn, (opt.H, opt.W))
            _, _, heatmap, des1 = fe.run(img, pts1)
            des1 = des1.T


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
    print('==> Finshed Demo.')

"""
Compute perfs on my kp with some mtf descriptor
"""

import os
import random
import time
import argparse
import numpy as np
import cv2

import tensorflow as tf

# my additions
import tools.cst as cst 
import tools.bench_tools as bench_tools
import tools.init_weights as init_weights
import tools.model_vgg as model
import methods.elf_vgg.tools as tools_elf

from utils.dump import saveh5, loadh5

parser = argparse.ArgumentParser()
parser.add_argument('--trials', type=str, default='100', help='xp id')
parser.add_argument('--lift_data_id', type=str, default='1', 
        help='dir that stores lift kp res/lift/"$kp_dir_id"/des')
parser.add_argument('--max_num_feat', type=int, default=1000,
        help='Maximum number of features.')
parser.add_argument('--h', type=int, default=480, help='new height')
parser.add_argument('--w', type=int, default=704, help='new width')
parser.add_argument('--resize', type=int, default=0, help='set to 1 to resize img')
parser.add_argument('--feat_name', type=str, default='b3', help='feat_name')
args = parser.parse_args()

new_size = (args.w, args.h)
norm = 'L2'

# lift kp are in this directory
i_des_dir = os.path.join('res/lift/', args.lift_data_id, 'des_no_aug')
v_des_dir = os.path.join('res/lift/', args.lift_data_id, 'des_aug')

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


with tf.Graph().as_default():

    img_op = tf.placeholder(dtype=tf.float32, shape=[1, new_size[1], new_size[0], 3])
    feats_op, _ = model.model_grad(img_op, is_training=False)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # load vgg proxy-descriptor    
        init_weights.restore_vgg(sess, '%s/vgg/data.ckpt'%cst.WEIGHT_DIR)


        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
            
            global_start_time = time.time()
            for scene_name in cst.SCENE_LIST:
                duration = time.time() - global_start_time
                print('*** %s *** %d:%02d'%(scene_name, duration/60, duration%60))
                f.write('*** %s *** %d:%02d\n'%(scene_name, duration/60, duration%60))

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

                # get 1st img, (resize it), convert to BW
                img0_fn = os.path.join(cst.DATA_DIR, scene_name,'%d.ppm'%1)
                #print('img_fn: %s'%img0_fn)
                img0 = cv2.imread(img0_fn)
                old_size0 = (img0.shape[1], img0.shape[0])
                if args.resize==1:
                    img0 = cv2.resize(img0, new_size, interpolation=cv2.INTER_LINEAR)
                

                # detect
                lift_out0_fn = os.path.join(lift_dir, scene_name, '%d.h5'%1)
                lift_out0 = loadh5(lift_out0_fn)
                
                # convert to cv2 kp for prototype conformity
                kp0 = []
                for line in lift_out0['keypoints'][:args.max_num_feat,:]:
                    kp = cv2.KeyPoint(x=line[0],y=line[1], _size=2,
                            _angle=0, _response=0, _octave=0, _class_id=0)
                    kp0.append(kp)

                # draw kp on img
                kp_on_img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
                kp_on_img0 = np.tile(np.expand_dims(kp_on_img0,2), (1,1,3))
                for i,kp in enumerate(kp0):
                    pt = (int(round(kp.pt[0])), int(round(kp.pt[1])))
                    cv2.circle(kp_on_img0, pt, 1, (0, 255, 0), -1, lineType=16)
                
                # convert to np array for interpolation
                pts0 = np.vstack([[kp.pt[0],kp.pt[1]] for kp in kp0])
                pts0 = pts0.T

                # description
                patch = tools_elf.preproc(img0) 
                des_coarse = sess.run(feats_op[args.feat_name], 
                        feed_dict={img_op: patch})[0,:,:,:]
                des0 = tools_elf.SuperPoint_interpolate(
                        pts0, des_coarse, new_size[0], new_size[1])


                for img_key in range(2,cst.MAX_IMG_NUM+1):
                    # get 2nd img, (resize it), convert to BW
                    img1_fn = os.path.join(cst.DATA_DIR, scene_name,'%d.ppm'%img_key)
                    #print('img_fn: %s'%img1_fn)
                    img1 = cv2.imread(img1_fn)
                    H = np.loadtxt(os.path.join(cst.DATA_DIR, scene_name, 'H_1_%d'%img_key))
                    
                    # rectify H
                    if args.resize==1:
                        s1x = 1.0*new_size[0]/old_size0[0]
                        s1y = 1.0*new_size[1]/old_size0[1]
                        six = 1.0*new_size[0]/img1.shape[1]
                        siy = 1.0*new_size[1]/img1.shape[0]
                        #print('s1x - s1y - six - siy', s1x, s1y, six, siy)
                        H = np.diag((six,siy,1)).dot(H.dot( np.diag((1.0/s1x, 1.0/s1y, 1)) ) )
                        img1 = cv2.resize(img1, new_size, interpolation=cv2.INTER_LINEAR)

                    # detection
                    lift_out1_fn = os.path.join(lift_dir, scene_name, '%d.h5'%img_key)
                    lift_out1 = loadh5(lift_out1_fn)
                    
                    # convert to cv2 kp for prototype conformity
                    kp1 = []
                    for line in lift_out1['keypoints'][:args.max_num_feat,:]:
                        kp = cv2.KeyPoint(x=line[0],y=line[1], _size=2,
                                _angle=0, _response=0, _octave=0, _class_id=0)
                        kp1.append(kp)


                    # draw kp on img
                    kp_on_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                    kp_on_img1 = np.tile(np.expand_dims(kp_on_img1,2), (1,1,3))
                    for i,kp in enumerate(kp1):
                        pt = (int(round(kp.pt[0])), int(round(kp.pt[1])))
                        cv2.circle(kp_on_img1, pt, 1, (0, 255, 0), -1, lineType=16)
                    
                    # convert to np array for interpolation
                    pts1 = np.vstack([[kp.pt[0],kp.pt[1]] for kp in kp1])
                    pts1 = pts1.T

                    # description
                    patch = tools_elf.preproc(img1) 
                    des_coarse = sess.run(feats_op[args.feat_name], 
                            feed_dict={img_op: patch})[0,:,:,:]
                    des1 = tools_elf.SuperPoint_interpolate(
                            pts1, des_coarse, new_size[0], new_size[1])


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



        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)
            f.close()

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


f.close()

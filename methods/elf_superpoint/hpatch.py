"""

"""

import os
import random
import time
import argparse
import numpy as np
import cv2

import tensorflow as tf

import tools.cst as cst
import tools.bench_tools as bench_tools
import tools.init_weights as init_weights
import methods.elf_vgg.tools as tools_elf

import tools.model_vgg as model_des
from tools.init_weights import restore_vgg as restore

import methods.elf_superpoint.model as model_det


W_NPY_DIR = 'meta/weights/superpoint/npy/'

parser = argparse.ArgumentParser()
parser.add_argument('--trials', type=str, default='100', help='xp id')
parser.add_argument('--nms_dist', type=int, default=4, help='Non Maximum Suppression (NMS) distance (default: 4).')
parser.add_argument('--border_remove', type=int, default=4, help='Border of the img where you ignore the kp')
parser.add_argument('--max_num_feat', type=int, default=1000,  help='Maximum number of features.')
parser.add_argument('--H', type=int, default=480, help='new height')
parser.add_argument('--W', type=int, default=640, help='new width')
parser.add_argument('--kp_size', type=int, default=4, help='mock kp size')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--grad_name', type=str, default='b3', help='grad_name')
parser.add_argument('--feat_name', type=str, default='b3', help='feat_name')
parser.add_argument('--thr_k_size', type=int, default=5, 
        help='kernel size for blur before thr threshold computation.')
parser.add_argument('--thr_sigma', type=int, default=5, 
        help='kernel size for blur before thr threshold computation.')
parser.add_argument('--noise_k_size', type=int, default=5, 
        help='kernel size for blur before thresholding.')
parser.add_argument('--noise_sigma', type=int, default=5, 
        help='kernel size for blur before thresholding.')
parser.add_argument('--grad_thresh', type=float, default=1000, 
        help='superpoint gradients have a horrible variance so you need to\
        manually get rid of te few very high values')
parser.add_argument('--resize', type=int, default=1, help='Set to 1 to resize.')
args = parser.parse_args()

new_size = (args.W, args.H)
norm = 'L2'



# setup output dir
res_dir = os.path.join('res/superpoint/', args.trials)
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

# write human readable logs
f = open(os.path.join(res_dir, 'log.txt'), 'w')
f.write('elf\n')
f.write('thresh_overlap: %d\n'%cst.THRESH_OVERLAP)
f.write('thresh_desc: %d\n'%cst.THRESH_DESC)
f.write('thr_k_size: %d - thr_sigma: %d\n'%(args.thr_k_size, args.thr_sigma))
f.write('noise_k_size: %d - noise_sigma: %d\n'%(args.noise_k_size, args.noise_sigma))


FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
matcher = cv2.FlannBasedMatcher(index_params,search_params)


var_v = np.loadtxt('meta/net_vars/superpoint_var.txt', dtype=str,
        delimiter=',')
var_dict = {} # mapping tensor name <-> pickle id to load
for i in range(var_v.shape[0]):
    var_dict[var_v[i,0]] = var_v[i,1]
#print('var_dict', var_dict)

superpoint_var_list = var_dict.keys()
#print('superpoint_var_list', superpoint_var_list)


with tf.Graph().as_default():
    img_op = tf.placeholder(dtype=tf.float32, shape=[1, args.H, args.W, 1])
    img_col_op = tf.placeholder(dtype=tf.float32, shape=[1, args.H, args.W, 3])
    _, grads_op = model_det.model_grad(img_op, is_training=False)
    feats_op, _ = model_des.model_grad(img_col_op, is_training=False)

    net_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # restore superpoint detector
        for var in net_var_list:
            if var.op.name in superpoint_var_list:
                #print(var.op.name)
                scope, dummy, var_type = var.op.name.split("/")
                var_id = var_dict[var.op.name]
                w = np.load('%s/%s.npy'%(W_NPY_DIR, var_id))
                if var_type=='kernel':
                    # batch_size, C, H, W -> batch_size, H, W, C
                    w = tf.transpose(w, (2,3,1,0))
                #print('scope: %s - var_type: %s - var_id: %s' %(scope, var_type, var_id), w.shape)
                sess.run(var.assign(w))
        print('restore OK')
        # restore vgg descriptor
        init_weights.restore_vgg(sess, 'meta/weights/vgg/data.ckpt')
        # restore OK


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
                
                
                # get 1st img, (resize it), convert to BW
                img0_fn = os.path.join(cst.DATA_DIR, scene_name,'%d.ppm'%1)
                #print('img_fn: %s'%img0_fn)
                img0 = cv2.imread(img0_fn)
                old_size0 = (img0.shape[1], img0.shape[0])
                if args.resize==1:
                    img0 = cv2.resize(img0, new_size, interpolation=cv2.INTER_LINEAR)
                if img0.ndim == 3 and img0.shape[-1] == 3:
                    img0_bw = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
                
               
                # detection
                patch = tools_elf.preproc(img0_bw)
                grad = sess.run(grads_op[args.grad_name], 
                                    feed_dict={img_op: patch})[0][0,:,:,:]
                pts_fail, pts0 = tools_elf.postproc(grad, args.noise_k_size,
                        args.noise_sigma, args.thr_k_size, args.thr_sigma, 
                        args.nms_dist, args.border_remove, args.max_num_feat)

                if pts_fail:
                    print('all the scene is screwed')
                    for img_key in range(2, cst.MAX_IMG_NUM+1):
                        f.write('** %d **\n'%img_key)
                        # ms
                        print('min_conf: %.5f - my_raw_ms:%.3f - N1:%d - N2:%d - M:%d - M_d:%d - inter:%d'
                                %(-1, 0, 0, 0, 0, 0, 0))
                        f.write('rep:%.3f - N1:%d - N2:%d - M:%d\n'%(0,0,0,0))
                        f.write('ms:%.3f - N1:%d - N2:%d - M:%d - M_d:%d - inter:%d\n'
                                %(0, 0, 0, 0, 0, 0))
                    print('goto next scene')
                    break # goto next scene

                # convert to cv2 kp for prototype homogeneity
                kp0 = []
                for pt in pts0.T:
                    kp = cv2.KeyPoint(x=pt[0],y=pt[1], _size=4,
                        _angle=0, _response=0, _octave=0, _class_id=0)
                    kp0.append(kp) 

                # draw kp on img
                kp_on_img0 = np.tile(np.expand_dims(img0_bw,2), (1,1,3))
                for i,kp in enumerate(kp0):
                    pt = (int(round(kp.pt[0])), int(round(kp.pt[1])))
                    cv2.circle(kp_on_img0, pt, 1, (0, 255, 0), -1, lineType=16)


                # descriptor
                patch = tools_elf.preproc(img0) 
                des_coarse = sess.run(feats_op[args.feat_name],
                        feed_dict={img_col_op: patch})[0,:,:,:]
                des0 = tools_elf.SuperPoint_interpolate(
                        pts0, des_coarse, new_size[0], new_size[1])


                for img_key in range(2,cst.MAX_IMG_NUM+1):

                    # get 2nd img
                    img1_fn = os.path.join(cst.DATA_DIR, scene_name,'%d.ppm'%img_key)
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
                    if img1.ndim == 3 and img1.shape[-1] == 3:
                        img1_bw = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                    kp_on_img1 = img1_bw.copy()


                    # detection
                    patch = tools_elf.preproc(img1_bw)
                    grad = sess.run(grads_op[args.grad_name], 
                            feed_dict={img_op: patch})[0][0,:,:,:]
                    pts_fail, pts1 = tools_elf.postproc(grad, args.noise_k_size, 
                            args.noise_sigma, args.thr_k_size, args.thr_sigma, 
                            args.nms_dist, args.border_remove, args.max_num_feat)
                    
                    if pts_fail:
                        f.write('** %d **\n'%img_key)
                        # ms
                        print('min_conf: %.5f - my_raw_ms:%.3f - N1:%d - N2:%d - M:%d - M_d:%d - inter:%d'
                                %(-1, 0, 0, 0, 0, 0, 0))
                        f.write('rep:%.3f - N1:%d - N2:%d - M:%d\n'%(0,0,0,0))
                        f.write('ms:%.3f - N1:%d - N2:%d - M:%d - M_d:%d - inter:%d\n'
                                %(0, 0, 0, 0, 0, 0))
                        continue # go to next img


                    # convert to cv2 kp for prototype homogeneity
                    kp1 = []
                    for pt in pts1.T:
                        kp = cv2.KeyPoint(x=pt[0],y=pt[1], _size=4,
                            _angle=0, _response=0, _octave=0, _class_id=0)
                        kp1.append(kp)
                    
                    # draw kp on img
                    kp_on_img1 = np.tile(np.expand_dims(img1_bw,2), (1,1,3))
                    for i,kp in enumerate(kp0):
                        pt = (int(round(kp.pt[0])), int(round(kp.pt[1])))
                        cv2.circle(kp_on_img1, pt, 1, (0, 255, 0), -1, lineType=16)

                    # descriptor
                    patch = tools_elf.preproc(img1) 
                    des_coarse = sess.run(feats_op[args.feat_name],
                            feed_dict={img_col_op: patch})[0,:,:,:]
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
                    
                        match_des_img = cv2.drawMatches(img0_bw, kp0, img1_bw, kp1, good, None, 
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



"""
Generate kp for tf-lift to compute their descriptors.
You need to set the scale of your kp
"""
import os
import random
import time
import argparse
import numpy as np
import cv2

import tensorflow as tf

import tools.cst as cst 

import tools.init_weights as init_weights
import tools.model_vgg as model
import methods.elf_vgg.tools as tools_elf

parser = argparse.ArgumentParser()
parser.add_argument('--trials', type=str, default='100', help='xp id')
parser.add_argument('--nms_dist', type=int, default=4, help='Non Maximum Suppression (NMS) distance (default: 4).')
parser.add_argument('--border_remove', type=int, default=4, help='Border of the img where you ignore the kp')
parser.add_argument('--max_num_feat', type=int, default=1000, help='Maximum number of features.')
parser.add_argument('--H', type=int, default=480, help='new height')
parser.add_argument('--W', type=int, default=640, help='new width')
parser.add_argument('--grad_name', type=str, default='pool2', help='grad_name')
parser.add_argument('--feat_name', type=str, default='pool4', help='feat_name')
parser.add_argument('--thr_k_size', type=int, default=5, help='kernel size for blur before otsu threshold computation.')
parser.add_argument('--thr_sigma', type=int, default=5, help='gaussian var. blur for threshold computation.')
parser.add_argument('--noise_k_size', type=int, default=5, help='kernel size blur before thresholding.')
parser.add_argument('--noise_sigma', type=int, default=5,  help='gaussian var. blur before thresholding.')
parser.add_argument('--model', type=str, default='vgg', help='net model in {alexnet, vgg, xception}')
parser.add_argument('--resize', type=int, default=0, help='set to 1 to resize img')
parser.add_argument('--save2txt', type=int, default=0, help='set to 1 to save kp to file txt')
args = parser.parse_args()

new_size = (args.W, args.H)
norm = 'L2'

# setup output dir
res_dir = os.path.join('res/elf/', args.trials)
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

with tf.Graph().as_default():

    img_op = tf.placeholder(dtype=tf.float32, shape=[1,new_size[1],new_size[0],3])
    _, grads_op = model.model_grad(img_op, is_training=False)
   

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
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

                out_dir = os.path.join(res_dir, scene_name)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                
                
                for img_key in range(1,cst.MAX_IMG_NUM+1):
                    # get 1st img, (resize it), convert to BW
                    img_fn = os.path.join(cst.DATA_DIR,scene_name,'%d.ppm'%img_key)
                    img = cv2.imread(img_fn)
                    if args.resize==1:
                        img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

                    
                    patch = tools_elf.preproc(img) 
                    grad = sess.run(grads_op[args.grad_name], 
                                feed_dict={img_op: patch})[0][0,:,:,:]
                    pts_fail, pts = tools_elf.postproc(grad, args.noise_k_size,
                            args.noise_sigma, args.thr_k_size, args.thr_sigma, 
                            args.nms_dist, args.border_remove, args.max_num_feat)


                    SCALE = 4.2
                    scale_v = np.ones(pts.shape[1])*SCALE
                    pts = np.vstack((pts, scale_v))
                    pts = pts[[0,1,3,2],:]
                    pts_fn = os.path.join(out_dir, '%d_kp.txt'%img_key)
                    np.savetxt(pts_fn, pts.T)

        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)




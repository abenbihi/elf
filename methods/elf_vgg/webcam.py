"""
kp: me with gradmap nms 
des: tentative descriptor 
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
args = parser.parse_args()

new_size = (args.W, args.H)
norm = 'L2'

if args.model=='alexnet':
    import tools.model_alexnet as model
    from tools.init_weights import restore_alexnet as restore

elif args.model=='vgg':
    import tools.model_vgg as model
    from tools.init_weights import restore_vgg as restore

elif args.model=='xception':
    import tools.model_xception as model
else:
    print('Error: this model is not available: %s'%args.model)
    exit(1)


# setup output dir
res_dir = os.path.join('res/elf/', args.trials)
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

# write human readable logs
f = open(os.path.join(res_dir, 'log.txt'), 'w')
f.write('elf\n')
f.write('thresh_overlap: %d\n'%cst.THRESH_OVERLAP)
f.write('thresh_desc: %d\n'%cst.THRESH_DESC)


FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
matcher = cv2.FlannBasedMatcher(index_params,search_params)


global_start_time = time.time()
H = np.eye(3)
with tf.Graph().as_default():
    
    # define network operations graph
    img_op = tf.placeholder(dtype=tf.float32, shape=[1, new_size[1], new_size[0], 3])
    feats_op, grads_op = model.model_grad(img_op, is_training=False)

    with tf.Session() as sess:
        # load trained models
        sess.run(tf.global_variables_initializer())
        if args.model=='vgg' or args.model=='alexnet':
            restore(sess, 'meta/weights/%s/data.ckpt'%args.model)
        elif args.model=='xception':
            log_dir = 'meta/weights/xception/'
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                print('No checkpoint file found in: %s' %(log_dir))
                exit(1)
        else:
            print('Error: this model is not available: %s'%args.model)
            exit(1)


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
     
                img_dir = os.path.join(cst.DATA_DIR, scene_name, 'test/image_color')
                img_list = os.listdir(img_dir)

                # get 1st img, (resize it), convert to BW
                img0_fn = os.path.join(img_dir, img_list[0])
                #print('img_fn: %s'%img0_fn)
                img0 = cv2.imread(img0_fn)
                if args.resize==1:
                    img0 = cv2.resize(img0, new_size, interpolation=cv2.INTER_LINEAR)

                
                # detection
                patch = tools_elf.preproc(img0) 
                grad = sess.run(grads_op[args.grad_name], 
                                feed_dict={img_op: patch})[0][0,:,:,:]
                pts_fail, pts0 = tools_elf.postproc(grad, args.noise_k_size, 
                        args.noise_sigma, args.thr_k_size, args.thr_sigma, 
                        args.nms_dist, args.border_remove, args.max_num_feat)

                if pts_fail:
                    print('all the scene is screwed')
                    for img_name in img_list[1:]:
                        if not '.png' in img_name:
                            continue
                        f.write('** %s **\n'%img_name)
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
                img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
                kp_on_img0 = np.tile(np.expand_dims(img0,2), (1,1,3))
                for i,kp in enumerate(kp0):
                    pt = (int(round(kp.pt[0])), int(round(kp.pt[1])))
                    cv2.circle(kp_on_img0, pt, 1, (0, 255, 0), -1, lineType=16)

                # description
                des_coarse = sess.run(feats_op[args.feat_name], 
                            feed_dict={img_op: patch})[0,:,:,:]
                des0 = tools_elf.SuperPoint_interpolate(
                        pts0, des_coarse, new_size[0], new_size[1])


                for img_name in img_list[1:]:
                    if not '.png' in img_name:
                        continue
                    # get 2nd img, (resize it), convert to BW
                    img1_fn = os.path.join(img_dir, img_name)
                    #print('img_fn: %s'%img1_fn)
                    img1 = cv2.imread(img1_fn)
                    if args.resize==1:
                        img1 = cv2.resize(img1, new_size, interpolation=cv2.INTER_LINEAR)


                    # detection
                    patch = tools_elf.preproc(img1)
                    grad = sess.run(grads_op[args.grad_name], 
                                    feed_dict={img_op: patch})[0][0,:,:,:]
                    
                    pts_fail, pts1 = tools_elf.postproc(grad, args.noise_k_size, 
                        args.noise_sigma, args.thr_k_size, args.thr_sigma, 
                        args.nms_dist, args.border_remove, args.max_num_feat)

                    if pts_fail:
                        f.write('** %s **\n'%img_name)
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
                    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                    kp_on_img1 = np.tile(np.expand_dims(img1,2), (1,1,3))
                    for i,kp in enumerate(kp0):
                        pt = (int(round(kp.pt[0])), int(round(kp.pt[1])))
                        cv2.circle(kp_on_img1, pt, 1, (0, 255, 0), -1, lineType=16)

                    # description
                    des_coarse = sess.run(feats_op[args.feat_name], 
                                feed_dict={img_op: patch})[0,:,:,:]
                    des1 = tools_elf.SuperPoint_interpolate(
                            pts1, des_coarse, new_size[0], new_size[1])


                    # metrics
                    print('** %s **'%img_name)
                    f.write('** %s **\n'%img_name)

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




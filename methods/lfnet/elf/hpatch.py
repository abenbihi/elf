"""
ELF-detection with lfnet. Proxy description with VGG.
"""

from __future__ import print_function
import os
import sys
import time
import importlib
import cv2
import pickle

import numpy as np

import tensorflow as tf

from inference import *

# my additions
import tools.cst as cst 
import tools.bench_tools as bench_tools

import tools.init_weights as init_weights
import tools.model_vgg as model_des
import methods.elf_vgg.tools as tools_elf
import methods.lfnet.elf.lfnet_desc as lfnet_desc


MODEL_PATH = '%s/models'%cst.LFNET_DIR
if MODEL_PATH not in sys.path:
    sys.path.append(MODEL_PATH)


def build_networks(config, photo, is_training):

    DET = importlib.import_module(config.detector)
    detector = DET.Model(config, is_training)

    if config.input_inst_norm:
        print('Apply instance norm on input photos')
        photos1 = instance_normalization(photo)
    
    # inference.py
    if config.use_nms3d:
        heatmaps, det_endpoints = build_multi_scale_deep_detector_3DNMS(config, detector, photo, reuse=False)
    else:
        heatmaps, det_endpoints = build_multi_scale_deep_detector(config, detector, photo, reuse=False)

    # extract patches
    kpts = det_endpoints['kpts']
    batch_inds = det_endpoints['batch_inds']
    
    kp_patches = build_patch_extraction(config, det_endpoints, photo)

    # Descriptor
    descriptor = lfnet_desc.Model(config, is_training)
    desc_endpoints = descriptor.build_model(photo, reuse=False) # [B*K,D]


    # elf backprop
    grads_op = desc_endpoints['grads_dict']
    feats_op = desc_endpoints['feats_dict']


    # scale and orientation (extra)
    scale_maps = det_endpoints['scale_maps']
    ori_maps = det_endpoints['ori_maps'] # cos/sin
    degree_maps, _ = get_degree_maps(ori_maps) # degree (rgb psuedo color code)
    kpts_scale = det_endpoints['kpts_scale']
    kpts_ori = det_endpoints['kpts_ori']
    kpts_ori = tf.atan2(kpts_ori[:,1], kpts_ori[:,0]) # radian

    ops = {
        'grads_dict': grads_op,
        'feats_dict': feats_op,
        'photo': photo,
        'is_training': is_training,
        'kpts': kpts,
        #'feats': desc_feats,
        # EXTRA
        'scale_maps': scale_maps,
        'kpts_scale': kpts_scale,
        'degree_maps': degree_maps,
        'kpts_ori': kpts_ori,
    }

    return ops


def main(config):

   
    # Build Networks
    tf.reset_default_graph()

    # detector
    photo_ph = tf.placeholder(tf.float32, [1, config.h, config.w, 1]) # input grayscale image, normalized by 0~1
    is_training = tf.constant(False) # Always False in testing

    ops = build_networks(config, photo_ph, is_training)
    lfnet_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    
    # proxy-descriptor
    img_col_op = tf.placeholder(dtype=tf.float32, shape=[1, config.h, config.w, 3])
    feats_op, _ = model_des.model_grad(img_col_op, is_training=False)

    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True 

    sess = tf.Session(config=tfconfig)
    sess.run(tf.global_variables_initializer())

    # load lfnet model
    saver = tf.train.Saver(lfnet_var_list)
    print('Load trained models...')
    if os.path.isdir(config.model):
        checkpoint = tf.train.latest_checkpoint(config.model)
        model_dir = config.model
    else:
        checkpoint = config.model
        model_dir = os.path.dirname(config.model)
    if checkpoint is not None:
        print('Checkpoint', os.path.basename(checkpoint))
        print("[{}] Resuming...".format(time.asctime()))
        saver.restore(sess, checkpoint)
    else:
        raise ValueError('Cannot load model from {}'.format(model_dir))    
    print('Done.')
       

    # load vgg proxy-descriptor    
    init_weights.restore_vgg(sess, '%s/vgg/data.ckpt'%cst.WEIGHT_DIR)


    avg_elapsed_time = 0
    ##########################################################################

    new_size = (config.w, config.h)

    # setup output dir
    res_dir = os.path.join('res/elf-lfnet/', config.trials)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
   
    # write human readable logs
    f = open(os.path.join(res_dir, 'log.txt'), 'w')
    f.write('grad_block: %d - grad_name: %s\n'%(config.grad_block, config.grad_name))
    f.write('feat_block: %d - feat_name: %s\n'%(config.feat_block, config.feat_name))
    f.write('thr_k_size: %d - ostu_sigma: %d\n'%(config.thr_k_size, config.thr_sigma))
    f.write('noise_k_size: %d - noise_sigma: %d\n'%(config.noise_k_size, config.noise_sigma))
    global_start_time = time.time()
    
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params,search_params)
    norm = 'L2'

    for scene_name in cst.SCENE_LIST:
        duration = time.time() - global_start_time
        print('*** %s *** %d:%02d'%(scene_name, duration/60, duration%60))
        f.write('*** %s *** %d:%02d\n'%(scene_name, duration/60, duration%60))

        
        # get 1st img, (resize it), convert to BW
        img0_fn = os.path.join(cst.DATA_DIR, scene_name,'%d.ppm'%1)
        img0 = cv2.imread(img0_fn)
        old_size0 = (img0.shape[1], img0.shape[0])
        if config.resize==1:
            img0 = cv2.resize(img0, new_size, interpolation=cv2.INTER_LINEAR)
        if img0.ndim == 3 and img0.shape[-1] == 3:
            img0_bw = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        kp_on_img0 = img0_bw.copy()


        # detection
        patch = tools_elf.preproc(img0_bw)
        block = config.grad_block
        grad_name = config.grad_name
        grad = sess.run(ops['grads_dict']['block-%d'%block][grad_name],
                feed_dict={photo_ph: patch})[0]
        grad = np.squeeze(grad)
        pts_fail, pts0 = tools_elf.postproc(grad, config.noise_k_size, 
                config.noise_sigma, config.thr_k_size, config.thr_sigma, 
                config.nms_dist, config.border_remove, config.max_num_feat)
            
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
            continue # goto next scene

        # convert to cv2 kp for prototype homogeneity
        kp0 = []
        for pt in pts0.T:
            kp = cv2.KeyPoint(x=pt[0],y=pt[1], _size=4,
                _angle=0, _response=0, _octave=0, _class_id=0)
            kp0.append(kp) 

        # draw kp on img
        kp_on_img0 = np.tile(np.expand_dims(kp_on_img0,2), (1,1,3))
        for i,kp in enumerate(kp0):
            pt = (int(round(kp.pt[0])), int(round(kp.pt[1])))
            cv2.circle(kp_on_img0, pt, 1, (0, 255, 0), -1, lineType=16)


        # descriptor
        patch = tools_elf.preproc(img0) 
        des_coarse = sess.run(feats_op[config.feat_name],
                feed_dict={img_col_op: patch})[0,:,:,:]
        des0 = tools_elf.SuperPoint_interpolate(
                pts0, des_coarse, new_size[0], new_size[1])


        for img_key in range(2,cst.MAX_IMG_NUM+1):
            # get 2nd img
            img1_fn = os.path.join(cst.DATA_DIR, scene_name,'%d.ppm'%img_key)
            img1 = cv2.imread(img1_fn)
            H = np.loadtxt(os.path.join(cst.DATA_DIR, scene_name, 'H_1_%d'%img_key))
            
            if config.resize==1:
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
            block = config.grad_block
            grad_name = config.grad_name
            grad = sess.run(ops['grads_dict']['block-%d'%block][grad_name],
                    feed_dict={photo_ph: patch})[0]
            grad = np.squeeze(grad)
            pts_fail, pts1 = tools_elf.postproc(grad, config.noise_k_size, 
                    config.noise_sigma, config.thr_k_size, config.thr_sigma, 
                    config.nms_dist, config.border_remove, config.max_num_feat)
            
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
            kp_on_img1 = np.tile(np.expand_dims(kp_on_img1,2), (1,1,3))
            for i,kp in enumerate(kp0):
                pt = (int(round(kp.pt[0])), int(round(kp.pt[1])))
                cv2.circle(kp_on_img1, pt, 1, (0, 255, 0), -1, lineType=16)


            # descriptor
            patch = tools_elf.preproc(img1) 
            des_coarse = sess.run(feats_op[config.feat_name],
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
            
                match_des_img = cv2.drawMatches(img0, kp0, img1, kp1, good, None, 
                        flags=2)
                cv2.imshow('match_des', match_des_img)
                cv2.imshow('kp_on', np.hstack((kp_on_img0, kp_on_img1)))
                cv2.waitKey(0)


    f.close()
    print('Done.')

if __name__ == '__main__':

    from common.argparse_utils import *
    parser = get_parser()

    general_arg = add_argument_group('General', parser)
    general_arg.add_argument('--num_threads', type=int, default=8,
                            help='the number of threads (for dataset)')

    io_arg = add_argument_group('In/Out', parser)
    io_arg.add_argument('--in_dir', type=str, default='./samples',
                            help='input image directory')
    # io_arg.add_argument('--in_dir', type=str, default='./release/outdoor_examples/images/sacre_coeur/dense/images',
    #                         help='input image directory')
    io_arg.add_argument('--out_dir', type=str, default='./dump_feats',
                            help='where to save keypoints')
    io_arg.add_argument('--full_output', type=str2bool, default=False,
                            help='dump keypoint image')

    model_arg = add_argument_group('Model', parser)
    model_arg.add_argument('--model', type=str, default='./release/models/outdoor/',
                            help='model file or directory')
    model_arg.add_argument('--top_k', type=int, default=500,
                            help='number of keypoints')
    model_arg.add_argument('--max_longer_edge', type=int, default=640,
                            help='resize image (do nothing if max_longer_edge <= 0)')

    
    misc_arg = add_argument_group('Misc.', parser)
    misc_arg.add_argument('--use_nms3d', type=str2bool, default=False,
                            help='use NMS3D to detect keypoints')

    misc_arg.add_argument('--trials', type=str, default='100', help='xp id')
    misc_arg.add_argument('--nms_dist', type=int, default=4,
            help='Non Maximum Suppression (NMS) distance (default: 4).')
    misc_arg.add_argument('--border_remove', type=int, default=4,
            help='Border of the img where you ignore the kp')
    misc_arg.add_argument('--max_num_feat', type=int, default=1000,
            help='Maximum number of features.')
    misc_arg.add_argument('--h', type=int, default=480, help='new height')
    misc_arg.add_argument('--w', type=int, default=640, help='new width')
    misc_arg.add_argument('--resize', type=int, default=1, help='Set to 1 to resize.')
    misc_arg.add_argument('--kp_size', type=int, default=4, help='mock kp size')
    misc_arg.add_argument('--grad_block', type=int, default=2, 
            help='resnet block from which to extract gradient')
    misc_arg.add_argument('--grad_name', type=str, default='conv2', 
            help='layer in the block for gradient')
    misc_arg.add_argument('--feat_block', type=int, default=2, 
            help='resnet block from which to extract descriptor')
    misc_arg.add_argument('--feat_name', type=str, default='conv2', 
            help='layer in the block for description')
    misc_arg.add_argument('--thr_k_size', type=int, default=5, 
            help='kernel size for blur before thr threshold computation.')
    misc_arg.add_argument('--thr_sigma', type=int, default=5,
            help='kernel size for blur before thr threshold computation.')
    misc_arg.add_argument('--noise_k_size', type=int, default=5, 
            help='kernel size for blur before thresholding.')
    misc_arg.add_argument('--noise_sigma', type=int, default=5, 
            help='kernel size for blur before thresholding.')

    tmp_config, unparsed = get_config(parser)

    if len(unparsed) > 0:
        raise ValueError('Miss finding argument: unparsed={}\n'.format(unparsed))

    # restore other hyperparams to build model
    if os.path.isdir(tmp_config.model):
        config_path = os.path.join(tmp_config.model, 'config.pkl')
    else:
        config_path = os.path.join(os.path.dirname(tmp_config.model), 'config.pkl')
    try:
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
    except:
        raise ValueError('Fail to open {}'.format(config_path))

    for attr, dst_val in sorted(vars(tmp_config).items()):
        if hasattr(config, attr):
            src_val = getattr(config, attr)
            if src_val != dst_val:
                setattr(config, attr, dst_val)
        else:
            setattr(config, attr, dst_val)

    main(config)

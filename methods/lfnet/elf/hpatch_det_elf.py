
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


MODEL_PATH = '%s/models'%cst.LFNET_DIR
if MODEL_PATH not in sys.path:
    sys.path.append(MODEL_PATH)

def make_patch(photo, pts):
    
    patch_l = []
    patch_size = 32
    ps_2 = int(patch_size/2) # half patch size
    img = np.squeeze(photo)
    #print('img.shape', img.shape)
    #h,w = photo.shape[1:3]
    h,w = img.shape
    for pt in pts:
        #patch 
        #print(pt)
        bi = int(np.maximum(0, pt[1]-ps_2)) # begin_i
        ei = int(np.minimum(h, pt[1]+ps_2+1)) # end_i
        bj = int(np.maximum(0, pt[0]-ps_2))
        ej = int(np.minimum(w, pt[0]+ps_2+1))
        #print(bi, ei, bj, ej)
        #patch = photo[0,bi:ei, bj:ej,:]
        patch = img[bi:ei, bj:ej]
        #print('patch_shape', patch.shape)
        if patch.shape[0]!=patch_size or patch.shape[1]!=patch_size:
            patch = cv2.resize(patch, (patch_size, patch_size),
                    interpolation=cv2.INTER_AREA)
        patch = np.expand_dims(patch, 2)
        patch_l.append(patch)
    #print(len(patch_l))
    
    if len(patch_l)==1:
        patch_l = np.array(patch_l)
        np.expand_dims(patch_l, 0)
    else:
        patch_l = np.array(patch_l)
    #print('patch_shape', patch_l.shape)

    patch_l = patch_l.astype(np.float32)
    #patch_l = np.expand_dims(patch_l, 0)
    #print('patch_shape', patch_l.shape)
    return patch_l


#def build_networks(config, photo, is_training):
def build_networks(config, photo, is_training, batch_inds,
        kpts_xy, kpts_scale, kpts_ori, kp_patches):

    print('config.detector', config.detector)
    num_kp_op       = tf.shape(kpts_xy)[0]
    DET = importlib.import_module(config.detector)
    detector = DET.Model(config, is_training)

    if config.input_inst_norm:
        print('Apply instance norm on input photos')
        photos1 = instance_normalization(photo)

    if config.use_nms3d:
        print('use_nms3d: YES')
        heatmaps, det_endpoints = build_multi_scale_deep_detector_3DNMS(config, detector, photo, reuse=False)
    else:
        print('use_nms3d: NO')
        heatmaps, det_endpoints = build_multi_scale_deep_detector(config, detector, photo, reuse=False)

    # extract patches
    #kpts = det_endpoints['kpts']
    #print('kpts', kpts)
    
    ####
    det_endpoints['kpts'] = kpts_xy
    det_endpoints['batch_inds'] = batch_inds
    det_endpoints['num_kpts'] = batch_inds
    det_endpoints['kpts_ori'] = kpts_ori
    det_endpoints['kpts_scale'] = kpts_scale

    #batch_inds = det_endpoints['batch_inds']
    #print('batch_inds', batch_inds)

    #kp_patches = build_patch_extraction(config, det_endpoints, photo)
    #print('kp_patches', kp_patches)

    # Descriptor
    DESC = importlib.import_module(config.descriptor)
    descriptor = DESC.Model(config, is_training)
    desc_feats, desc_endpoints = descriptor.build_model(kp_patches, reuse=False) # [B*K,D]

    # scale and orientation (extra)
    #scale_maps = det_endpoints['scale_maps']
    #ori_maps = det_endpoints['ori_maps'] # cos/sin
    #degree_maps, _ = get_degree_maps(ori_maps) # degree (rgb psuedo color code)
    #kpts_scale = det_endpoints['kpts_scale']
    #kpts_ori = det_endpoints['kpts_ori']
    #kpts_ori = tf.atan2(kpts_ori[:,1], kpts_ori[:,0]) # radian

    ops = {
        'photo': photo,
        'is_training': is_training,
        'kpts': kpts_xy,
        'feats': desc_feats,
        # EXTRA
        #'scale_maps': scale_maps,
        #'kpts_scale': kpts_scale,
        #'degree_maps': degree_maps,
        #'kpts_ori': kpts_ori,
    }

    return ops

def main(config):

    # Build Networks
    tf.reset_default_graph()

    photo_ph = tf.placeholder(tf.float32, [1, None, None, 1]) # input grayscale image, normalized by 0~1
    is_training = tf.constant(False) # Always False in testing
    batch_inds_op   = tf.placeholder(tf.int32, [None]) # [B*K]
    kpts_xy_op      = tf.placeholder(tf.float32, [None, 2]) # [B*K, 2] x,y unnormalized coords
    kpts_scale_op   = tf.placeholder(tf.float32, [None]) # [B*K]
    kpts_ori_op     = tf.placeholder(tf.float32, [None,2]) #[B*K,2]
    kp_patches_op     = tf.placeholder(tf.float32, [None,32,32,1]) #[B*K,2]

    ops = build_networks(config, photo_ph, is_training, batch_inds_op,
            kpts_xy_op, kpts_scale_op, kpts_ori_op, kp_patches_op)

    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True 
    sess = tf.Session(config=tfconfig)
    sess.run(tf.global_variables_initializer())

    # load model
    saver = tf.train.Saver()
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

    avg_elapsed_time = 0
    
    
    ##########################################################################

    new_size = (config.w, config.h)
    
    kp_dir = 'res/elf/%s'%config.kp_dir_id

    # setup output dir
    res_dir = os.path.join('res/lfnet/', config.trials)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    # write human readable logs
    f = open(os.path.join(res_dir, 'log.txt'), 'w')
    f.write('lfnet\n')
    f.write('data: %s\n'%cst.DATA)
    f.write('thresh_overlap: %d\n'%cst.THRESH_OVERLAP)
    f.write('thresh_desc: %d\n'%cst.THRESH_DESC)

    
    # feature matcher handler for visualization
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params,search_params)
    norm = 'L2'

    
    global_start_time = time.time()
    for scene_name in cst.SCENE_LIST:
        duration = time.time() - global_start_time
        print('*** %s *** %d:%02d'%(scene_name, duration/60, duration%60))
        f.write('*** %s *** %d:%02d\n'%(scene_name, duration/60, duration%60))


        # get 1st img, (resize it), convert to BW
        img0_fn = os.path.join(cst.DATA_DIR, scene_name,'%d.ppm'%1)
        print('img_fn: %s'%img0_fn)
        img0 = cv2.imread(img0_fn)
        old_size0 = (img0.shape[1], img0.shape[0])
        if config.resize==1:
            img0 = cv2.resize(img0, new_size, interpolation=cv2.INTER_LINEAR)
        height, width = img0.shape[:2]
        if img0.ndim == 3 and img0.shape[-1] == 3:
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        img0_bw = img0.copy()
        img0 = img0[None,...,None].astype(np.float32) / 255.0 # normalize 0-1
        assert img0.ndim == 4 # [1,H,W,1]
        
        
        # elf-detection with mock scale and orientation
        SCALE = 0.5
        kp_fn = os.path.join(kp_dir, scene_name, '%d_kp.txt'%1)
        #in_kpts_xy = np.loadtxt(kp_fn).T[:,:2]
        in_kpts_xy = np.loadtxt(kp_fn)[:,:2]
        #print('in_kpts_xy.shape', in_kpts_xy.shape)
        in_kpts_ori = np.zeros((in_kpts_xy.shape[0],2))
        #print('in_kpts_ori.shape', in_kpts_ori.shape)
        in_kpts_scale = np.ones(in_kpts_xy.shape[0])*SCALE
        #print('in_kpts_scale.shape', in_kpts_scale.shape)
        in_batch_inds = np.arange(in_kpts_xy.shape[0])
        #print('in_batch_inds.shape', in_batch_inds.shape)
        in_kp_patches = make_patch(img0, in_kpts_xy)
        
        feed_dict = {
            photo_ph: img0,
            batch_inds_op: in_batch_inds,
            kpts_xy_op: in_kpts_xy, 
            kpts_scale_op: in_kpts_scale,
            kpts_ori_op: in_kpts_ori,
            kp_patches_op: in_kp_patches
        }

        # Dump keypoint locations and their features
        fetch_dict = {
            'kpts': ops['kpts'],
            'feats': ops['feats'],
        }
        outs = sess.run(fetch_dict, feed_dict=feed_dict)
        
        #print(outs['kpts'].shape)
        #print(outs['feats'].shape)
        pts0 = outs['kpts'].T
        #print('pts',pts)
        #print('pts.shape', pts.shape)
        #print('scene_name: %s - img_key: %d'%(scene_name, img_key))
        #input('wait')
        des0 = outs['feats']


        # convert to cv2 kp for prototype conformity
        kp0 = []
        for pt in pts0.T:
            kp = cv2.KeyPoint(x=pt[0],y=pt[1], _size=8,
                _angle=0, _response=0, _octave=0, _class_id=0)
            kp0.append(kp)

        # draw kp on img
        kp_on_img0 = np.tile(np.expand_dims(img0_bw,2), (1,1,3))
        for i,kp in enumerate(kp0):
            pt = (int(round(kp.pt[0])), int(round(kp.pt[1])))
            cv2.circle(kp_on_img0, pt, 1, (0, 255, 0), -1, lineType=16)


        for img_key in range(2,cst.MAX_IMG_NUM+1):

            # get 2nd img
            img1_fn = os.path.join(cst.DATA_DIR, scene_name,'%d.ppm'%img_key)
            #print('img_fn: %s'%img1_fn)
            img1 = cv2.imread(img1_fn)
            H = np.loadtxt(os.path.join(cst.DATA_DIR, scene_name, 'H_1_%d'%img_key))
            # correct H with new size
            if config.resize==1:
                s1x = 1.0*new_size[0]/old_size0[0]
                s1y = 1.0*new_size[1]/old_size0[1]
                six = 1.0*new_size[0]/img1.shape[1]
                siy = 1.0*new_size[1]/img1.shape[0]
                #print('s1x - s1y - six - siy', s1x, s1y, six, siy)
                H = np.diag((six,siy,1)).dot(H.dot( np.diag((1.0/s1x, 1.0/s1y, 1)) ) )
                img1 = cv2.resize(img1, new_size, interpolation=cv2.INTER_LINEAR)
            
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img1_bw = img1.copy()
            if img1.ndim == 3 and img1.shape[-1] == 3:
                img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            img1 = img1[None,...,None].astype(np.float32) / 255.0 # normalize 0-1
            assert img1.ndim == 4 # [1,H,W,1]


            # elf-detection with mock scale and orientation
            kp_fn = os.path.join(kp_dir, scene_name, '%d_kp.txt'%img_key)
            in_kpts_xy = np.loadtxt(kp_fn)[:,:2]
            #print('in_kpts_xy.shape', in_kpts_xy.shape)
            in_kpts_ori = np.zeros((in_kpts_xy.shape[0],2))
            #print('in_kpts_ori.shape', in_kpts_ori.shape)
            in_kpts_scale = np.ones(in_kpts_xy.shape[0])*SCALE
            #print('in_kpts_scale.shape', in_kpts_scale.shape)
            in_batch_inds = np.arange(in_kpts_xy.shape[0])
            #print('in_batch_inds.shape', in_batch_inds.shape)
            in_kp_patches = make_patch(img1, in_kpts_xy)
            
            feed_dict = {
                photo_ph: img1,
                batch_inds_op: in_batch_inds,
                kpts_xy_op: in_kpts_xy, 
                kpts_scale_op: in_kpts_scale,
                kpts_ori_op: in_kpts_ori,
                kp_patches_op: in_kp_patches
            }

            # Dump keypoint locations and their features
            fetch_dict = {
                'kpts': ops['kpts'],
                'feats': ops['feats'],
            }
            outs = sess.run(fetch_dict, feed_dict=feed_dict)
            
            pts1 = outs['kpts'].T
            des1 = outs['feats']

            # convert to cv2 kp for prototype conformity
            kp1 = []
            for pt in pts1.T:
                kp = cv2.KeyPoint(x=pt[0],y=pt[1], _size=2,
                    _angle=0, _response=0, _octave=0, _class_id=0)
                kp1.append(kp)


            # draw kp
            kp_on_img1 = np.tile(np.expand_dims(img1_bw,2), (1,1,3))
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
                matches = matcher.knnMatch(des0, des1,k=2)
                for i,(m,n) in enumerate(matches):
                    if m.distance < 0.8*n.distance:
                        good.append(m)

                match_des_img = cv2.drawMatches(img0_bw, kp0, img1_bw, kp1, good, None, 
                        flags=2)
                cv2.imshow('match_des', match_des_img)
                cv2.imshow('kp_on', np.hstack((kp_on_img0, kp_on_img1)))
                cv2.waitKey(0)


    f.close()
    print('Done.')

if __name__ == '__main__':

    from common.argparse_utils import *
    parser = get_parser()

    parser.add_argument('--trials', type=str)
    parser.add_argument('--h', type=int)
    parser.add_argument('--w', type=int)
    parser.add_argument('--resize', type=int)
    parser.add_argument('--kp_dir_id', type=str, help='elf kp are stored in res/elf/kp_dir_id')


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

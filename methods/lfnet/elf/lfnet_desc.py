# -*- coding: utf-8 -*-

import math
import tensorflow as tf
from common.tf_layer_utils import *
from common.tf_train_utils import get_activation_fn
from det_tools import instance_normalization


def get_model(inputs, is_training, 
              out_dim=128,
              init_num_channels=64,
              num_conv_layers=3,
              conv_ksize=3,
              activation_fn=tf.nn.relu,
              use_xavier=True, perform_bn=True,
              bn_trainable=True,
              bn_decay=None, bn_affine=True, use_bias=True,
              feat_norm='l2norm',
              reuse=False, name='SimpleDesc'):

    channels_list = [init_num_channels * 2**i for i in range(num_conv_layers)]
    print('===== {} (reuse={}) ====='.format(name, reuse))

    grads_dict, feats_dict = {},{}

    with tf.variable_scope(name, reuse=reuse) as net_sc:
        curr_in = inputs

        for i, num_channels in enumerate(channels_list):
            grads_dict['block-%d'%(i+1)] = {}
            feats_dict['block-%d'%(i+1)] = {}

            curr_in = conv2d(curr_in, num_channels,
                        kernel_size=conv_ksize, scope='conv{}'.format(i+1),
                        stride=2, padding='SAME',
                        #stride=1, padding='SAME',
                        use_xavier=use_xavier, use_bias=use_bias)

            #curr_in
            print('block-%d - conv.shape'%(i+1), curr_in.get_shape())
            grads_dict['block-%d'%(i+1)]['conv'] = tf.gradients(
                    curr_in, inputs, curr_in)
            feats_dict['block-%d'%(i+1)]['conv'] = curr_in

            ## bn
            #curr_in = batch_norm_act(curr_in, activation_fn, 
            curr_in = batch_norm_act(curr_in, None, # fuck relu
                                     perform_bn=perform_bn,
                                     is_training=is_training, 
                                     bn_decay=bn_decay,
                                     bn_affine=bn_affine,
                                     bnname='bn{}'.format(i+1)
                                    )
            grads_dict['block-%d'%(i+1)]['bn'] = tf.gradients(
                    curr_in, inputs, curr_in)
            feats_dict['block-%d'%(i+1)]['bn'] = curr_in


            # pool
            #curr_in = tf.nn.max_pool(curr_in, ksize=[1,2,2,1], strides=[1,2,2,1],
            #        padding='VALID', name='pool')
            #grads_dict['block-%d'%(i+1)]['pool'] = tf.gradients(
            #        curr_in, inputs, curr_in)
            #feats_dict['block-%d'%(i+1)]['pool'] = curr_in
            #print('block-%d - pool.shape'%(i+1), curr_in.get_shape())

            #print('#{} conv-bn-act {}'.format(i+1, curr_in.shape))
        
        endpoints = {}
        endpoints['grads_dict'] = grads_dict
        endpoints['feats_dict'] = feats_dict

        return  endpoints


class Model(object):
    def __init__(self, config, is_training):
        self.config = config
        self.is_training = is_training
        self.activation_fn = get_activation_fn(config.desc_activ_fn, **{'alpha': config.desc_leaky_alpha})

    def build_model(self, feat_maps, reuse=False, name='SimpleDesc'):
        # out_dim = getattr(self.config, 'desc_dim', 128)
        endpoints = get_model(feat_maps, self.is_training,
                        out_dim=self.config.desc_dim,
                        init_num_channels=self.config.desc_net_channel,
                        num_conv_layers=self.config.desc_net_depth,
                        conv_ksize=self.config.desc_conv_ksize,
                        activation_fn=self.activation_fn,
                        perform_bn=self.config.desc_perform_bn,
                        feat_norm=self.config.desc_norm,
                        reuse=reuse, name=name)
        
        return endpoints 

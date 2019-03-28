"""A binary to train using a single GPU.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os.path
import math
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib.framework.python.framework import checkpoint_utils

def restore_vgg(sess, path):
    caffe_weights_fn = path 
    if sys.version_info[0] >= 3:
        caffe_weights = np.load(caffe_weights_fn, encoding = 'latin1').item() # fuck pickles
    else:
        caffe_weights = np.load(caffe_weights_fn).item()

    #caffe_weights = np.load(caffe_weights_fn).item()
    var_to_restore = []
    vgg_var_list = [l.split("\n")[0] for l in open('meta/net_vars/vgg_var.txt').readlines()]
    var_list =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    for var in var_list:
        if var.op.name in vgg_var_list:
            var_to_restore.append(var)
            #print(var.op.name)
            if len(var.op.name.split("/"))!=3:
                print('No load %s' %var.op.name)
                continue
            scope, dummy, var_type = var.op.name.split("/")
            #print('scope: %s, var_type: %s' %(scope, var_type))
            if var_type=='kernel':
                sess.run(var.assign(caffe_weights[scope]['weights']))
            else:
                sess.run(var.assign(caffe_weights[scope]['biases']))

def restore_alexnet(sess, path):
    print('Weights initialization ...')
    caffe_weights_fn = path 
    caffe_weights = np.load(caffe_weights_fn).item()
    var_to_restore = []
    nosplit_list = [l.split("\n")[0] for l in open('meta/net_vars/nosplit_var.txt').readlines()]
    split_list = [l.split("\n")[0] for l in open('meta/net_vars/split_var.txt').readlines()]
    var_list =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    for var in var_list:
        if len(var.op.name.split("/"))!=3:
            #print('No load %s' %var.op.name)
            continue

        if var.op.name in nosplit_list:
            var_to_restore.append(var)
            #print(var.op.name)
            scope, dummy, var_type = var.op.name.split("/")
            #print('scope: %s, var_type: %s' %(scope, var_type))
            if var_type=='kernel':
                sess.run(var.assign(caffe_weights[scope]['weights']))
            else:
                sess.run(var.assign(caffe_weights[scope]['biases']))
        
        if var.op.name in split_list:
            var_to_restore.append(var)
            #print(var.op.name)
            scope, dummy, var_type = var.op.name.split("/")
            scope_root = scope[:-2]
            split = int(scope[-1])
            #print('\nscope: %s, var_type: %s' %(scope, var_type))
            if var_type=='kernel':
                if split==1:
                    begin = 0
                    end = int(caffe_weights[scope_root]['weights'].shape[3]/2)
                    #print('%d -> %d' %(begin, end))
                    #print(caffe_weights[scope_root]['weights'].shape)
                    w = caffe_weights[scope_root]['weights'][:,:,:,0:end]
                else:
                    begin = int(caffe_weights[scope_root]['weights'].shape[3]/2)
                    #print('%d ->' %(begin))
                    #print(caffe_weights[scope_root]['weights'].shape)
                    w = caffe_weights[scope_root]['weights'][:,:,:,begin:]
                sess.run(var.assign(w))
            else:
                if split==1:
                    begin = 0
                    end = int(caffe_weights[scope_root]['biases'].shape[0]/2)
                    #print('%d -> %d' %(begin, end))
                    #print(caffe_weights[scope_root]['biases'].shape)
                    b = caffe_weights[scope_root]['biases'][begin:end]
                else:
                    begin = int(caffe_weights[scope_root]['biases'].shape[0]/2)
                    #print('%d ->' %(begin))
                    #print(caffe_weights[scope_root]['biases'].shape)
                    b = caffe_weights[scope_root]['biases'][begin:]
                sess.run(var.assign(b))
    print('Weights initialization Done')



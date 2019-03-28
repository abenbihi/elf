
"""
- netmork model
- loss
- optimizer
- summary
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile


from six.moves import urllib
import tensorflow as tf
from math import sqrt
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import array_ops

FLAGS = tf.app.flags.FLAGS


@ops.RegisterGradient("MaxPoolGradWithArgmax")
def _MaxPoolGradGradWithArgmax(op, grad):
  print(len(op.outputs))
  print(len(op.inputs))
  print(op.name)
  return (array_ops.zeros(
      shape=array_ops.shape(op.inputs[0]),
      dtype=op.inputs[0].dtype), array_ops.zeros(
          shape=array_ops.shape(op.inputs[1]), dtype=op.inputs[1].dtype),
          gen_nn_ops._max_pool_grad_grad_with_argmax(
              op.inputs[0],
              grad,
              op.inputs[2],
              op.get_attr("ksize"),
              op.get_attr("strides"),
              padding=op.get_attr("padding")))
              

def model_grad(images, is_training, reuse=False):
    """
    Args:
      images: Images returned from distorted_inputs() or inputs().
    Returns:
      Logits.
    """
    
    print('inference::input', images.get_shape())
    bn = True
    lrn = False
    grads_dict = {}
    feat = {}
 
    with tf.variable_scope('conv1', reuse=reuse) as scope: #1
        conv1 = tf.layers.conv2d(inputs=images, filters=96, kernel_size=(11,11), strides=(4,4),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv1 = tf.contrib.layers.batch_norm(conv1, fused=True, decay=0.9, is_training=is_training)
        if lrn:
            conv1 = tf.nn.local_response_normalization(conv1,depth_radius=2,alpha=1.99999994948e-05,beta=0.75,bias=1.0,name='norm1')
        conv1 = tf.nn.relu(conv1)
        print('conv1', conv1.get_shape())
        pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name='pool') #pool1
        print('pool1', pool1.get_shape())
 
    pool1_1, pool1_2 = tf.split(pool1, 2, 3)
    print('pool1_1.shape: ', pool1_1.get_shape())
    print('pool1_2.shape: ', pool1_2.get_shape())
    
    with tf.variable_scope('conv2_1', reuse=reuse) as scope:#3
        conv2_1 = tf.layers.conv2d(inputs=pool1_1, filters=128, kernel_size=(5, 5),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv2_1 = tf.nn.relu(conv2_1)
        print('conv2_1', conv2_1.get_shape())
        
    with tf.variable_scope('conv2_2', reuse=reuse) as scope:#3
        conv2_2 = tf.layers.conv2d(inputs=pool1_2, filters=128, kernel_size=(5, 5),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv2_2 = tf.nn.relu(conv2_2)
        print('conv2_2', conv2_2.get_shape())
        
    conv2 = tf.concat([conv2_1, conv2_2], 3)
    if bn:
        conv2 = tf.contrib.layers.batch_norm(conv2,fused=True, decay=0.9, is_training=is_training)
    if lrn:
        conv2 = tf.nn.local_response_normalization(conv2,depth_radius=2,alpha=1.99999994948e-05,beta=0.75,bias=1.0,name='norm2')
    pool2 = tf.nn.max_pool(conv2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool2
    print('pool2', pool2.get_shape())
  
    with tf.variable_scope('conv3', reuse=reuse) as scope:#5
        conv3 = tf.layers.conv2d(inputs=pool2, filters=384, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv3 = tf.contrib.layers.batch_norm(conv3, fused=True, decay=0.9, is_training=is_training)
        conv3 = tf.nn.relu(conv3)
        print('conv3', conv3.get_shape())

    # useless, for my usage
    pool3 = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool3') 
    
    conv3_1, conv3_2 = tf.split(conv3, 2, 3)
    print('conv3_1.shape: ', conv3_1.get_shape())
    print('conv3_2.shape: ', conv3_2.get_shape())

    with tf.variable_scope('conv4_1', reuse=reuse) as scope:
        conv4_1 = tf.layers.conv2d(inputs=conv3_1, filters=192, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv4_1 = tf.contrib.layers.batch_norm(conv4_1,fused=True, decay=0.9, is_training=is_training)
        conv4_1 = tf.nn.relu(conv4_1)
        print('conv4_1', conv4_1.get_shape())

    with tf.variable_scope('conv4_2', reuse=reuse) as scope:
        conv4_2 = tf.layers.conv2d(inputs=conv3_2, filters=192, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv4_2 = tf.contrib.layers.batch_norm(conv4_2,fused=True, decay=0.9, is_training=is_training)
        conv4_2 = tf.nn.relu(conv4_2)
        print('conv4_2', conv4_2.get_shape())
    
    # useless, just for my usage
    conv4 = tf.concat([conv4_1, conv4_2], 3) 
    pool4 = tf.nn.max_pool(conv4, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name='pool4')
    
    with tf.variable_scope('conv5_1', reuse=reuse) as scope:
        conv5_1 = tf.layers.conv2d(inputs=conv4_1, filters=128, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv5_1 = tf.nn.relu(conv5_1)
        print('conv5_1', conv5_1.get_shape())

    with tf.variable_scope('conv5_2', reuse=reuse) as scope:
        conv5_2 = tf.layers.conv2d(inputs=conv4_2, filters=128, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv5_2 = tf.nn.relu(conv5_2)
        print('conv5_2', conv5_2.get_shape())
    
    conv5 = tf.concat([conv5_1, conv5_2], 3)
    if bn:
        conv5 = tf.contrib.layers.batch_norm(conv5,fused=True,decay=0.9,is_training=is_training)

    pool5 = tf.nn.max_pool(conv5, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name='pool')
    print('pool5', pool5.get_shape())
     
    
    # features % images
    #grads_dict['conv1'] = tf.gradients(conv1, images, conv1)
    grads_dict['pool1'] = tf.gradients(pool1, images, pool1)
    #grads_dict['conv2'] = tf.gradients(conv2, images, conv2)
    grads_dict['pool2'] = tf.gradients(pool2, images, pool2)
    #grads_dict['conv3'] = tf.gradients(conv3, images, conv3)
    #grads_dict['conv4'] = tf.gradients(conv4, images, conv4)
    #grads_dict['conv5'] = tf.gradients(conv5, images, conv5)
    #grads_dict['pool5'] = tf.gradients(pool5, images, pool5)
    #grads_dict['fc6'] = tf.gradients(fc6, images, fc6)
    #grads_dict['fc7'] = tf.gradients(fc7, images, fc7)
    #grads_dict['fc8'] = tf.gradients(fc8, images, fc8)

    feat['pool1'] = pool1
    feat['pool2'] = pool2
    feat['pool3'] = pool3
    feat['pool4'] = pool4
    feat['pool5'] = pool5
    return feat,  grads_dict


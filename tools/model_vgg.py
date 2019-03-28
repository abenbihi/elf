
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
              
def model(images, is_training, reuse=False):
    """ Network model
    Args:
      images: Images returned from distorted_inputs() or inputs().
    Returns:
      Logits.
    """
    
    print('inference::input', images.get_shape())
    bn = False
    relu = False
    feat, grads_dict = {},{}
 
    with tf.variable_scope('conv1_1', reuse=reuse) as scope: #1
        conv1_1 = tf.layers.conv2d(inputs=images, filters=64, kernel_size=(3,3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv1_1 = tf.contrib.layers.batch_norm(conv1_1, fused=True, decay=0.9, is_training=is_training)
        if relu:
            conv1_1 = tf.nn.relu(conv1_1)
        print('conv1_1', conv1_1.get_shape())
  
    with tf.variable_scope('conv1_2', reuse=reuse) as scope: #2
        conv1_2 = tf.layers.conv2d(inputs=conv1_1, filters=64, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv1_2 = tf.contrib.layers.batch_norm(conv1_2, fused=True, decay=0.9, is_training=is_training)
        if relu:
            conv1_2 = tf.nn.relu(conv1_2)
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool1
        #feat['pool1'] = pool1
        print('pool1', pool1.get_shape())
  
    with tf.variable_scope('conv2_1', reuse=reuse) as scope:#3
        conv2_1 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv2_1 = tf.contrib.layers.batch_norm(conv2_1, fused=True, decay=0.9, is_training=is_training)
        if relu:
            conv2_1 = tf.nn.relu(conv2_1)
        print('conv2_1', conv2_1.get_shape())

    with tf.variable_scope('conv2_2', reuse=reuse) as scope:#4
        conv2_2 = tf.layers.conv2d(inputs=conv2_1, filters=128, kernel_size=(3, 3),
            padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv2_2 = tf.contrib.layers.batch_norm(conv2_2, fused=True, decay=0.9, is_training=is_training)
        if relu:
            conv2_2 = tf.nn.relu(conv2_2)
        pool2 = tf.nn.max_pool(conv2_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool2
        #feat['pool2'] = pool2
        print('pool2', pool2.get_shape())
  
    with tf.variable_scope('conv3_1', reuse=reuse) as scope:#5
        conv3_1 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv3_1 = tf.contrib.layers.batch_norm(conv3_1, fused=True, decay=0.9, is_training=is_training)
        if relu:
            conv3_1 = tf.nn.relu(conv3_1)
        print('conv3_1', conv3_1.get_shape())

    with tf.variable_scope('conv3_2', reuse=reuse) as scope:#6
        conv3_2 = tf.layers.conv2d(inputs=conv3_1, filters=256, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv3_2 = tf.contrib.layers.batch_norm(conv3_2, fused=True, decay=0.9, is_training=is_training)
        if relu:
            conv3_2 = tf.nn.relu(conv3_2)
        print('conv3_2', conv3_2.get_shape())

    with tf.variable_scope('conv3_3', reuse=reuse) as scope:#7
        conv3_3 = tf.layers.conv2d(inputs=conv3_2, filters=256, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv3_3 = tf.contrib.layers.batch_norm(conv3_3, fused=True, decay=0.9, is_training=is_training)
        if relu:
            conv3_3 = tf.nn.relu(conv3_3)
        pool3 = tf.nn.max_pool(conv3_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool3
        #pool3 = tf.nn.l2_normalize(pool3, dim=3, epsilon=1e-12)
        feat['pool3'] = pool3
        print('pool3', pool3.get_shape())

    with tf.variable_scope('conv4_1', reuse=reuse) as scope:# 8
        conv4_1 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv4_1 = tf.contrib.layers.batch_norm(conv4_1, fused=True, decay=0.9, is_training=is_training)
        if relu:
            conv4_1 = tf.nn.relu(conv4_1)
        print('conv4_1', conv4_1.get_shape())

    with tf.variable_scope('conv4_2', reuse=reuse) as scope:#9
        conv4_2 = tf.layers.conv2d(inputs=conv4_1, filters=512, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv4_2 = tf.contrib.layers.batch_norm(conv4_2, fused=True, decay=0.9, is_training=is_training)
        if relu:
            conv4_2 = tf.nn.relu(conv4_2)
        print('conv4_2', conv4_2.get_shape())

    with tf.variable_scope('conv4_3', reuse=reuse) as scope:#10
        conv4_3 = tf.layers.conv2d(inputs=conv4_2, filters=512, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv4_3 = tf.contrib.layers.batch_norm(conv4_3, fused=True, decay=0.9, is_training=is_training)
        if relu:
            conv4_3 = tf.nn.relu(conv4_3)
        #conv4_3 = tf.nn.l2_normalize(conv4_3, dim=3, epsilon=1e-12)
        #feat['conv4_3'] = conv4_3
        pool4 = tf.nn.max_pool(conv4_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool4
        #pool4 = tf.nn.l2_normalize(pool4, dim=3, epsilon=1e-12)
        #feat['pool4'] = pool4
        print('pool4', pool4.get_shape())

    with tf.variable_scope('conv5_1', reuse=reuse) as scope:#11
        conv5_1 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv5_1 = tf.contrib.layers.batch_norm(conv5_1, fused=True, decay=0.9, is_training=is_training)
        if relu:
            conv5_1 = tf.nn.relu(conv5_1)
        print('conv5_1', conv5_1.get_shape())
    
    with tf.variable_scope('conv5_2', reuse=reuse) as scope:#12
        conv5_2 = tf.layers.conv2d(inputs=conv5_1, filters=512, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv5_2 = tf.contrib.layers.batch_norm(conv5_2, fused=True, decay=0.9, is_training=is_training)
        if relu:
            conv5_2 = tf.nn.relu(conv5_2)
        print('conv5_2', conv5_2.get_shape())

    with tf.variable_scope('conv5_3', reuse=reuse) as scope:#13
        conv5_3 = tf.layers.conv2d(inputs=conv5_2, filters=512, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv5_3 = tf.contrib.layers.batch_norm(conv5_3, fused=True, decay=0.9, is_training=is_training)
        if relu:
            conv5_3 = tf.nn.relu(conv5_3)
        pool5 = tf.nn.max_pool(conv5_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool5
        #pool5 = tf.nn.l2_normalize(pool5, dim=3, epsilon=1e-12)
        print('pool5', pool5.get_shape())
    
    # yayayaya
    #feat['pool1'] = tf.gradients(pool1, conv1_2, pool1)
    #feat['pool2'] = tf.gradients(pool2, conv2_2, pool2)
    #feat['pool3'] = tf.gradients(pool3, conv3_3, pool3)
    #feat['pool4'] = tf.gradients(pool4, conv4_3, pool4)
    #feat['pool5'] = tf.gradients(pool5, conv5_3, pool5)
    
    # grad prop from 5
    #grads_dict['pool5'] = tf.gradients(pool5, conv5_3, pool5)
    #grads_dict['pool4'] = tf.gradients(pool4, conv4_3, grads_dict['pool5'])
    #grads_dict['pool3'] = tf.gradients(pool3, conv3_3, grads_dict['pool4'])
    #grads_dict['pool2'] = tf.gradients(pool2, conv2_2, grads_dict['pool3'])
    ##grads_dict['pool1'] = tf.gradients(pool1, conv1_2, grads_dict['pool2'])
    #grads_dict['pool1'] = tf.gradients(pool1, images, grads_dict['pool2'])
    feat['conv1_1'] = conv1_1
    feat['conv1_2'] = conv1_2
    feat['pool1'] = pool1
    feat['conv2_1'] = conv2_1
    feat['conv2_2'] = conv2_2
    feat['pool2'] = pool2
    feat['conv3_1'] = conv3_1
    feat['conv3_2'] = conv3_2
    feat['conv3_3'] = conv3_3
    feat['pool3'] = pool3
    feat['conv4_1'] = conv4_1
    feat['conv4_2'] = conv4_2
    feat['conv4_3'] = conv4_3
    feat['pool4'] = pool4
    feat['conv5_1'] = conv5_1
    feat['conv5_2'] = conv5_2
    feat['conv5_3'] = conv5_3
    feat['pool5'] = pool5
    return feat, grads_dict


def model_grad(images, is_training=False, reuse=False):
    """ Network model
    Args:
      images: [batch)size, H, W, C]
      is_training: True if traning mode (for batchnorm)
    """
    
    if not reuse:
        print('inference::input', images.get_shape())
    bn = False
    relu = True
    grads_dict = {}
    argmax = {}
    feat = {}
 
    with tf.variable_scope('conv1_1', reuse=reuse) as scope: #1
        conv1_1 = tf.layers.conv2d(inputs=images, filters=64, kernel_size=(3,3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv1_1 = tf.contrib.layers.batch_norm(conv1_1, fused=True, decay=0.9, is_training=is_training)
        if relu:
            conv1_1 = tf.nn.relu(conv1_1)
        if not reuse:
            print('conv1_1', conv1_1.get_shape())
        ##grads_dict['conv1_1'] = tf.gradients(conv1_1, images, conv1_1)
  
    with tf.variable_scope('conv1_2', reuse=reuse) as scope: #2
        conv1_2 = tf.layers.conv2d(inputs=conv1_1, filters=64, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv1_2 = tf.contrib.layers.batch_norm(conv1_2, fused=True, decay=0.9, is_training=is_training)
        if relu:
            conv1_2 = tf.nn.relu(conv1_2)
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool1
        #pool1,argmax['pool1'] = tf.nn.max_pool_with_argmax(conv1_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool1
        if not reuse:
            print('pool1', pool1.get_shape())
        #grads_dict['conv1_2'] = tf.gradients(conv1_2, images, conv1_2)
        #grads_dict['pool1'] = tf.gradients(pool1, conv1_2, pool1)
        #feat['pool1'] = pool1
  
    with tf.variable_scope('conv2_1', reuse=reuse) as scope:#3
        conv2_1 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv2_1 = tf.contrib.layers.batch_norm(conv2_1, fused=True, decay=0.9, is_training=is_training)
        if relu:
            conv2_1 = tf.nn.relu(conv2_1)
        if not reuse:
            print('conv2_1', conv2_1.get_shape())
        #grads_dict['conv2_1'] = tf.gradients(conv2_1, images, conv2_1)

    with tf.variable_scope('conv2_2', reuse=reuse) as scope:#4
        conv2_2 = tf.layers.conv2d(inputs=conv2_1, filters=128, kernel_size=(3, 3),
            padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv2_2 = tf.contrib.layers.batch_norm(conv2_2, fused=True, decay=0.9, is_training=is_training)
        if relu:
            conv2_2 = tf.nn.relu(conv2_2)
        pool2 = tf.nn.max_pool(conv2_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool2
        #pool2, argmax['pool2'] = tf.nn.max_pool_with_argmax(conv2_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool2
        if not reuse:
            print('pool2', pool2.get_shape())
        #grads_dict['conv2_2'] = tf.gradients(conv2_2, images, conv2_2)
        #grads_dict['pool2'] = tf.gradients(pool2, images, pool2)
  
    with tf.variable_scope('conv3_1', reuse=reuse) as scope:#5
        conv3_1 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv3_1 = tf.contrib.layers.batch_norm(conv3_1, fused=True, decay=0.9, is_training=is_training)
        if relu:
            conv3_1 = tf.nn.relu(conv3_1)
        if not reuse:
            print('conv3_1', conv3_1.get_shape())
        #grads_dict['conv3_1'] = tf.gradients(conv3_1, images, conv3_1)

    with tf.variable_scope('conv3_2', reuse=reuse) as scope:#6
        conv3_2 = tf.layers.conv2d(inputs=conv3_1, filters=256, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv3_2 = tf.contrib.layers.batch_norm(conv3_2, fused=True, decay=0.9, is_training=is_training)
        if relu:
            conv3_2 = tf.nn.relu(conv3_2)
        if not reuse:
            print('conv3_2', conv3_2.get_shape())
        #grads_dict['conv3_2'] = tf.gradients(conv3_2, images, conv3_2)

    with tf.variable_scope('conv3_3', reuse=reuse) as scope:#7
        conv3_3 = tf.layers.conv2d(inputs=conv3_2, filters=256, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv3_3 = tf.contrib.layers.batch_norm(conv3_3, fused=True, decay=0.9, is_training=is_training)
        if relu:
            conv3_3 = tf.nn.relu(conv3_3)
        pool3 = tf.nn.max_pool(conv3_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool3
        #pool3, argmax['pool3'] = tf.nn.max_pool_with_argmax(conv3_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool3
        if not reuse:
            print('pool3', pool3.get_shape())
        #grads_dict['conv3_3'] = tf.gradients(conv3_3, images, conv3_3)
        #grads_dict['pool3'] = tf.gradients(pool3, images, pool3)

    with tf.variable_scope('conv4_1', reuse=reuse) as scope:# 8
        conv4_1 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv4_1 = tf.contrib.layers.batch_norm(conv4_1, fused=True, decay=0.9, is_training=is_training)
        if relu:
            conv4_1 = tf.nn.relu(conv4_1)
        if not reuse:
            print('conv4_1', conv4_1.get_shape())
        #grads_dict['conv4_1'] = tf.gradients(conv4_1, images, conv4_1)

    with tf.variable_scope('conv4_2', reuse=reuse) as scope:#9
        conv4_2 = tf.layers.conv2d(inputs=conv4_1, filters=512, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv4_2 = tf.contrib.layers.batch_norm(conv4_2, fused=True, decay=0.9, is_training=is_training)
        if relu:
            conv4_2 = tf.nn.relu(conv4_2)
        if not reuse:
            print('conv4_2', conv4_2.get_shape())
        #grads_dict['conv4_2'] = tf.gradients(conv4_2, images, conv4_2)

    with tf.variable_scope('conv4_3', reuse=reuse) as scope:#10
        conv4_3 = tf.layers.conv2d(inputs=conv4_2, filters=512, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv4_3 = tf.contrib.layers.batch_norm(conv4_3, fused=True, decay=0.9, is_training=is_training)
        if relu:
            conv4_3 = tf.nn.relu(conv4_3)
        pool4 = tf.nn.max_pool(conv4_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool4
        #pool4, argmax['pool4'] = tf.nn.max_pool_with_argmax(conv4_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool4
        if not reuse:
            print('pool4', pool4.get_shape())
        #grads_dict['conv4_3'] = tf.gradients(conv4_3, images, conv4_3)
        #grads_dict['pool4'] = tf.gradients(pool4, images, pool4)

    with tf.variable_scope('conv5_1', reuse=reuse) as scope:#11
        conv5_1 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv5_1 = tf.contrib.layers.batch_norm(conv5_1, fused=True, decay=0.9, is_training=is_training)
        if relu:
            conv5_1 = tf.nn.relu(conv5_1)
        if not reuse:
            print('conv5_1', conv5_1.get_shape())
        #grads_dict['conv5_1'] = tf.gradients(conv5_1, images, conv5_1)
    
    with tf.variable_scope('conv5_2', reuse=reuse) as scope:#12
        conv5_2 = tf.layers.conv2d(inputs=conv5_1, filters=512, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv5_2 = tf.contrib.layers.batch_norm(conv5_2, fused=True, decay=0.9, is_training=is_training)
        if relu:
            conv5_2 = tf.nn.relu(conv5_2)
        if not reuse:
            print('conv5_2', conv5_2.get_shape())
        #grads_dict['conv5_2'] = tf.gradients(conv5_2, images, conv5_2)

    with tf.variable_scope('conv5_3', reuse=reuse) as scope:#13
        conv5_3 = tf.layers.conv2d(inputs=conv5_2, filters=512, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv5_3 = tf.contrib.layers.batch_norm(conv5_3, fused=True, decay=0.9, is_training=is_training)
        if relu:
            conv5_3 = tf.nn.relu(conv5_3)
        pool5 = tf.nn.max_pool(conv5_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool5
        #pool5, argmax['pool5'] = tf.nn.max_pool_with_argmax(conv5_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool5
        if not reuse:
            print('pool5', pool5.get_shape())
        #grads_dict['conv5_3'] = tf.gradients(conv5_3, images, conv5_3)
        #grads_dict['pool5'] = tf.gradients(pool5, images, pool5)
  
    # trials/0
    #grads_dict['pool1'] = tf.gradients(pool1, images)
    #grads_dict['pool2'] = tf.gradients(pool2, images)
    #grads_dict['pool3'] = tf.gradients(pool3, images)
    #grads_dict['pool4'] = tf.gradients(pool4, images)
    #grads_dict['pool5'] = tf.gradients(pool5, images)
    
    SOBEL = (0==1)
    if SOBEL:
        #NEW_C = 64 # pool1
        NEW_C = 128 # pool2
        #NEW_C = 512 # pool25
        sobel_x = tf.tile(sobel_x, (1,1,1,NEW_C))
        sobel_y = tf.tile(sobel_y, (1,1,1,NEW_C))


    #grads_dict['conv1_1'] = tf.gradients(conv1_1, images, conv1_1)
    #grads_dict['conv1_2'] = tf.gradients(conv1_2, images, conv1_2)
    
    #grads_dict['pool1_sobel_x'] = tf.gradients(pool1, images, sobel_x)
    #grads_dict['pool1_sobel_y'] = tf.gradients(pool1, images, sobel_y)

    #grads_dict['conv2_1'] = tf.gradients(conv2_1, images, conv2_1)
    #grads_dict['conv2_2'] = tf.gradients(conv2_2, images, conv2_2)
    
    if SOBEL:
        grads_dict['pool2_sobel_x'] = tf.gradients(pool2, images, sobel_x)
        grads_dict['pool2_sobel_y'] = tf.gradients(pool2, images, sobel_y)
        pool2 = tf.nn.l2_normalize(pool2, dim=3, epsilon=1e-12)
        sobel_x = tf.nn.l2_normalize(sobel_x, dim=3, epsilon=1e-12)
        sobel_y = tf.nn.l2_normalize(sobel_y, dim=3, epsilon=1e-12)
        grads_dict['pool2_min_sobel_x'] = tf.gradients(pool2, images, (pool2-sobel_x))
        grads_dict['pool2_min_sobel_y'] = tf.gradients(pool2, images, (pool2-sobel_y))

    
    # grad
    #grads_dict['pool1'] = tf.gradients(pool1, images, pool1)
    grads_dict['pool2'] = tf.gradients(pool2, images, pool2)
    #grads_dict['pool3'] = tf.gradients(pool3, images, pool3)
    #grads_dict['pool4'] = tf.gradients(pool4, images, pool4)
    #grads_dict['pool5'] = tf.gradients(pool5, images, pool5)
    
    
    #grads_dict['pool5_sobel_x'] = tf.gradients(pool5, images, sobel_x)
    #grads_dict['pool5_sobel_y'] = tf.gradients(pool5, images, sobel_y)
    #grads_dict['pool5_min_sobel_x'] = tf.gradients(pool5, images, (pool5-sobel_x))
    #grads_dict['pool5_min_sobel_y'] = tf.gradients(pool5, images, (pool5-sobel_y))

    # trials/1
    #grads_dict['conv1_1'] = tf.gradients(conv1_1, images, conv1_1)
    #print(grads_dict['conv1_1'][0].get_shape())
    #grads_dict['2_conv1_1'] = tf.gradients(
    #        #grads_dict['conv1_1'], images, images)
    #        grads_dict['conv1_1'], images,grads_dict['conv1_1'])
    
    #grads_dict['conv1_2'] = tf.gradients(conv1_2, images, conv1_2)
    #grads_dict['2_conv1_2'] = tf.gradients(
    #        grads_dict['conv1_2'], images,grads_dict['conv1_2'])
    #grads_dict['pool1'] = tf.gradients(pool1, images, pool1)
    #grads_dict['2_pool1'] = tf.gradients(
    #        grads_dict['pool1'], images, grads_dict['pool1'])
    #grads_dict['pool2'] = tf.gradients(pool2, images, pool2)
    #grads_dict['pool3'] = tf.gradients(pool3, images, pool3)
    #grads_dict['pool4'] = tf.gradients(pool4, images, pool4)
    #grads_dict['pool5'] = tf.gradients(pool5, images, pool5)

    # 
    #grads_dict['pool1_conv'] = tf.gradients(pool1, conv1_2)
    #grads_dict['pool2_conv'] = tf.gradients(pool2, conv2_2)
    #grads_dict['pool3_conv'] = tf.gradients(pool3, conv3_3)
    #grads_dict['pool4_conv'] = tf.gradients(pool4, conv4_3)
    #grads_dict['pool5_conv'] = tf.gradients(pool5, conv5_3)

    # trials/1
    # yayayaya
    #grads_dict['pool1_conv'] = tf.gradients(pool1, conv1_2, pool1)
    #grads_dict['pool2_conv'] = tf.gradients(pool2, conv2_2, pool2)
    #grads_dict['pool3_conv'] = tf.gradients(pool3, conv3_3, pool3)
    #grads_dict['pool4_conv'] = tf.gradients(pool4, conv4_3, pool4)
    #grads_dict['pool5_conv'] = tf.gradients(pool5, conv5_3, pool5)

    # grad prop from 5
    #grads_dict['pool5'] = tf.gradients(pool5, conv5_3, pool5)
    #grads_dict['pool4'] = tf.gradients(pool4, conv4_3, grads_dict['pool5'])
    #grads_dict['pool3'] = tf.gradients(pool3, conv3_3, grads_dict['pool4'])
    #grads_dict['pool2'] = tf.gradients(pool2, conv2_2, grads_dict['pool3'])
    ##grads_dict['pool1'] = tf.gradients(pool1, conv1_2, grads_dict['pool2'])
    #grads_dict['pool1'] = tf.gradients(pool1, images, grads_dict['pool2'])

    # fail if you prop it not from 5
    #grads_dict['pool5'] = tf.gradients(pool5, conv5_3, pool5)
    #grads_dict['pool4'] = tf.gradients(pool4, conv4_3, pool4)
    #grads_dict['pool3'] = tf.gradients(pool3, conv3_3, grads_dict['pool4'])
    #grads_dict['pool2'] = tf.gradients(pool2, conv2_2, grads_dict['pool3'])
    ##grads_dict['pool1'] = tf.gradients(pool1, conv1_2, grads_dict['pool2'])
    #grads_dict['pool1'] = tf.gradients(pool1, images, grads_dict['pool2'])

    #grads_dict['pool1_conv'] = tf.gradients(pool1, conv1_1, pool1)
    #grads_dict['pool2_conv'] = tf.gradients(pool2, conv2_1, pool2)
    #grads_dict['pool3_conv'] = tf.gradients(pool3, conv3_2, pool3)
    #grads_dict['pool4_conv'] = tf.gradients(pool4, conv4_2, pool4)
    #grads_dict['pool5_conv'] = tf.gradients(pool5, conv5_2, pool5)


    #grads_dict['pool1_conv'] = tf.gradients(pool1, images, pool1)
    #grads_dict['pool2_conv'] = tf.gradients(pool2, pool1, pool2)
    #grads_dict['pool3_conv'] = tf.gradients(pool3, pool2, pool3)
    #grads_dict['pool4_conv'] = tf.gradients(pool4, pool3, pool4)
    #grads_dict['pool5_conv'] = tf.gradients(pool5, pool4, pool5)

    # trials/2
    #grads_dict['pool1'] = tf.gradients(pool1, conv1_2, pool1)
    #grads_dict['pool2'] = tf.gradients(pool2, conv1_2, pool2)
    #grads_dict['pool3'] = tf.gradients(pool3, conv1_2, pool3)
    #grads_dict['pool4'] = tf.gradients(pool4, conv1_2, pool4)
    #grads_dict['pool5'] = tf.gradients(pool5, conv1_2, pool5)
    
    # trials/3 
    #grads_dict['pool1'] = tf.gradients(pool1, conv1_2)
    #grads_dict['pool2'] = tf.gradients(pool2, conv1_2)
    #grads_dict['pool3'] = tf.gradients(pool3, conv1_2)
    #grads_dict['pool4'] = tf.gradients(pool4, conv1_2)
    #grads_dict['pool5'] = tf.gradients(pool5, conv1_2)

    #pool4 = tf.nn.l2_normalize(pool4, dim=3, epsilon=1e-12)
    
    feat['pool1'] = pool1
    feat['pool2'] = pool2
    feat['pool3'] = pool3
    #feat['pool3'] = tf.nn.l2_normalize(feat['pool3'], dim=3, epsilon=1e-12)
    #feat['conv3_3'] = conv3_3
    feat['pool4'] = pool4
    #feat['pool4'] = tf.nn.l2_normalize(feat['pool4'], dim=3, epsilon=1e-12)
    feat['pool5'] = pool5

    # grad as feature trial 1
    #feat['pool5'] = tf.gradients(pool5, pool4, pool5)
    #feat['pool4'] = tf.gradients(pool4, pool3, pool4)
    #feat['pool4'] = tf.nn.l2_normalize(feat['pool4'], dim=3, epsilon=1e-12)
    #feat['pool3'] = tf.gradients(pool5, pool3, pool5)
    #feat['pool3'] = tf.gradients(pool3, pool2, pool3)

    return feat, grads_dict


def small_model_grad(images, sobel_x, sobel_y, is_training=False, reuse=False):
    """ Network model
    Args:
      images: [batch)size, H, W, C]
      is_training: True if traning mode (for batchnorm)
    """
    
    if not reuse:
        print('inference::input', images.get_shape())
    bn = False
    grads_dict = {}
    argmax = {}
    feat = {}
 
    with tf.variable_scope('conv1_1', reuse=reuse) as scope: #1
        conv1_1 = tf.layers.conv2d(inputs=images, filters=64, kernel_size=(3,3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv1_1 = tf.contrib.layers.batch_norm(conv1_1, fused=True, decay=0.9, is_training=is_training)
        conv1_1 = tf.nn.relu(conv1_1)
        if not reuse:
            print('conv1_1', conv1_1.get_shape())
        ##grads_dict['conv1_1'] = tf.gradients(conv1_1, images, conv1_1)
  
    with tf.variable_scope('conv1_2', reuse=reuse) as scope: #2
        conv1_2 = tf.layers.conv2d(inputs=conv1_1, filters=64, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv1_2 = tf.contrib.layers.batch_norm(conv1_2, fused=True, decay=0.9, is_training=is_training)
        conv1_2 = tf.nn.relu(conv1_2)
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool1
        #pool1,argmax['pool1'] = tf.nn.max_pool_with_argmax(conv1_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool1
        if not reuse:
            print('pool1', pool1.get_shape())
        #grads_dict['conv1_2'] = tf.gradients(conv1_2, images, conv1_2)
        #grads_dict['pool1'] = tf.gradients(pool1, conv1_2, pool1)
        #feat['pool1'] = pool1
  
    with tf.variable_scope('conv2_1', reuse=reuse) as scope:#3
        conv2_1 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv2_1 = tf.contrib.layers.batch_norm(conv2_1, fused=True, decay=0.9, is_training=is_training)
        conv2_1 = tf.nn.relu(conv2_1)
        if not reuse:
            print('conv2_1', conv2_1.get_shape())
        #grads_dict['conv2_1'] = tf.gradients(conv2_1, images, conv2_1)

    with tf.variable_scope('conv2_2', reuse=reuse) as scope:#4
        conv2_2 = tf.layers.conv2d(inputs=conv2_1, filters=128, kernel_size=(3, 3),
            padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv2_2 = tf.contrib.layers.batch_norm(conv2_2, fused=True, decay=0.9, is_training=is_training)
        conv2_2 = tf.nn.relu(conv2_2)
        pool2 = tf.nn.max_pool(conv2_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool2
        #pool2, argmax['pool2'] = tf.nn.max_pool_with_argmax(conv2_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool2
        if not reuse:
            print('pool2', pool2.get_shape())
        #grads_dict['conv2_2'] = tf.gradients(conv2_2, images, conv2_2)
        #grads_dict['pool2'] = tf.gradients(pool2, images, pool2)
  
    #with tf.variable_scope('conv3_1', reuse=reuse) as scope:#5
    #    conv3_1 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=(3, 3),
    #            padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
    #    if bn:
    #        conv3_1 = tf.contrib.layers.batch_norm(conv3_1, fused=True, decay=0.9, is_training=is_training)
    #    conv3_1 = tf.nn.relu(conv3_1)
    #    if not reuse:
    #        print('conv3_1', conv3_1.get_shape())
    #    #grads_dict['conv3_1'] = tf.gradients(conv3_1, images, conv3_1)

    #with tf.variable_scope('conv3_2', reuse=reuse) as scope:#6
    #    conv3_2 = tf.layers.conv2d(inputs=conv3_1, filters=256, kernel_size=(3, 3),
    #            padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
    #    if bn:
    #        conv3_2 = tf.contrib.layers.batch_norm(conv3_2, fused=True, decay=0.9, is_training=is_training)
    #    conv3_2 = tf.nn.relu(conv3_2)
    #    if not reuse:
    #        print('conv3_2', conv3_2.get_shape())
    #    #grads_dict['conv3_2'] = tf.gradients(conv3_2, images, conv3_2)

    #with tf.variable_scope('conv3_3', reuse=reuse) as scope:#7
    #    conv3_3 = tf.layers.conv2d(inputs=conv3_2, filters=256, kernel_size=(3, 3),
    #            padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
    #    if bn:
    #        conv3_3 = tf.contrib.layers.batch_norm(conv3_3, fused=True, decay=0.9, is_training=is_training)
    #    conv3_3 = tf.nn.relu(conv3_3)
    #    pool3 = tf.nn.max_pool(conv3_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool3
    #    #pool3, argmax['pool3'] = tf.nn.max_pool_with_argmax(conv3_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool3
    #    if not reuse:
    #        print('pool3', pool3.get_shape())
    #    #grads_dict['conv3_3'] = tf.gradients(conv3_3, images, conv3_3)
    #    #grads_dict['pool3'] = tf.gradients(pool3, images, pool3)

    #with tf.variable_scope('conv4_1', reuse=reuse) as scope:# 8
    #    conv4_1 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=(3, 3),
    #            padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
    #    if bn:
    #        conv4_1 = tf.contrib.layers.batch_norm(conv4_1, fused=True, decay=0.9, is_training=is_training)
    #    conv4_1 = tf.nn.relu(conv4_1)
    #    if not reuse:
    #        print('conv4_1', conv4_1.get_shape())
    #    #grads_dict['conv4_1'] = tf.gradients(conv4_1, images, conv4_1)

    #with tf.variable_scope('conv4_2', reuse=reuse) as scope:#9
    #    conv4_2 = tf.layers.conv2d(inputs=conv4_1, filters=512, kernel_size=(3, 3),
    #            padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
    #    if bn:
    #        conv4_2 = tf.contrib.layers.batch_norm(conv4_2, fused=True, decay=0.9, is_training=is_training)
    #    conv4_2 = tf.nn.relu(conv4_2)
    #    if not reuse:
    #        print('conv4_2', conv4_2.get_shape())
    #    #grads_dict['conv4_2'] = tf.gradients(conv4_2, images, conv4_2)

    #with tf.variable_scope('conv4_3', reuse=reuse) as scope:#10
    #    conv4_3 = tf.layers.conv2d(inputs=conv4_2, filters=512, kernel_size=(3, 3),
    #            padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
    #    if bn:
    #        conv4_3 = tf.contrib.layers.batch_norm(conv4_3, fused=True, decay=0.9, is_training=is_training)
    #    conv4_3 = tf.nn.relu(conv4_3)
    #    pool4 = tf.nn.max_pool(conv4_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool4
    #    #pool4, argmax['pool4'] = tf.nn.max_pool_with_argmax(conv4_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool4
    #    if not reuse:
    #        print('pool4', pool4.get_shape())
    #    #grads_dict['conv4_3'] = tf.gradients(conv4_3, images, conv4_3)
    #    #grads_dict['pool4'] = tf.gradients(pool4, images, pool4)

    #with tf.variable_scope('conv5_1', reuse=reuse) as scope:#11
    #    conv5_1 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=(3, 3),
    #            padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
    #    if bn:
    #        conv5_1 = tf.contrib.layers.batch_norm(conv5_1, fused=True, decay=0.9, is_training=is_training)
    #    conv5_1 = tf.nn.relu(conv5_1)
    #    if not reuse:
    #        print('conv5_1', conv5_1.get_shape())
    #    #grads_dict['conv5_1'] = tf.gradients(conv5_1, images, conv5_1)
    #
    #with tf.variable_scope('conv5_2', reuse=reuse) as scope:#12
    #    conv5_2 = tf.layers.conv2d(inputs=conv5_1, filters=512, kernel_size=(3, 3),
    #            padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
    #    if bn:
    #        conv5_2 = tf.contrib.layers.batch_norm(conv5_2, fused=True, decay=0.9, is_training=is_training)
    #    conv5_2 = tf.nn.relu(conv5_2)
    #    if not reuse:
    #        print('conv5_2', conv5_2.get_shape())
    #    #grads_dict['conv5_2'] = tf.gradients(conv5_2, images, conv5_2)

    #with tf.variable_scope('conv5_3', reuse=reuse) as scope:#13
    #    conv5_3 = tf.layers.conv2d(inputs=conv5_2, filters=512, kernel_size=(3, 3),
    #            padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
    #    if bn:
    #        conv5_3 = tf.contrib.layers.batch_norm(conv5_3, fused=True, decay=0.9, is_training=is_training)
    #    conv5_3 = tf.nn.relu(conv5_3)
    #    pool5 = tf.nn.max_pool(conv5_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool5
    #    #pool5, argmax['pool5'] = tf.nn.max_pool_with_argmax(conv5_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool5
    #    if not reuse:
    #        print('pool5', pool5.get_shape())
    #    #grads_dict['conv5_3'] = tf.gradients(conv5_3, images, conv5_3)
    #    #grads_dict['pool5'] = tf.gradients(pool5, images, pool5)
  
    # trials/0
    #grads_dict['pool1'] = tf.gradients(pool1, images)
    #grads_dict['pool2'] = tf.gradients(pool2, images)
    #grads_dict['pool3'] = tf.gradients(pool3, images)
    #grads_dict['pool4'] = tf.gradients(pool4, images)
    #grads_dict['pool5'] = tf.gradients(pool5, images)
    
    SOBEL = (1==1)
    if SOBEL:
        #NEW_C = 64 # pool1
        NEW_C = 128 # pool2
        #NEW_C = 512 # pool25
        sobel_x = tf.tile(sobel_x, (1,1,1,NEW_C))
        sobel_y = tf.tile(sobel_y, (1,1,1,NEW_C))


    #grads_dict['conv1_1'] = tf.gradients(conv1_1, images, conv1_1)
    #grads_dict['conv1_2'] = tf.gradients(conv1_2, images, conv1_2)
    
    grads_dict['pool1'] = tf.gradients(pool1, images, pool1)
    #grads_dict['pool1_sobel_x'] = tf.gradients(pool1, images, sobel_x)
    #grads_dict['pool1_sobel_y'] = tf.gradients(pool1, images, sobel_y)

    #grads_dict['conv2_1'] = tf.gradients(conv2_1, images, conv2_1)
    #grads_dict['conv2_2'] = tf.gradients(conv2_2, images, conv2_2)
    
    grads_dict['pool2'] = tf.gradients(pool2, images, pool2)
    #grads_dict['pool3'] = tf.gradients(pool3, images, pool3)
    if SOBEL:
        grads_dict['pool2_sobel_x'] = tf.gradients(pool2, images, sobel_x)
        grads_dict['pool2_sobel_y'] = tf.gradients(pool2, images, sobel_y)
        pool2 = tf.nn.l2_normalize(pool2, dim=3, epsilon=1e-12)
        sobel_x = tf.nn.l2_normalize(sobel_x, dim=3, epsilon=1e-12)
        sobel_y = tf.nn.l2_normalize(sobel_y, dim=3, epsilon=1e-12)
        grads_dict['pool2_min_sobel_x'] = tf.gradients(pool2, images, (pool2-sobel_x))
        grads_dict['pool2_min_sobel_y'] = tf.gradients(pool2, images, (pool2-sobel_y))


    #grads_dict['pool3'] = tf.gradients(pool3, images, pool3)
    #grads_dict['pool4'] = tf.gradients(pool4, images, pool4)
    #grads_dict['pool5'] = tf.gradients(pool5, images, pool5)
    #grads_dict['pool5_sobel_x'] = tf.gradients(pool5, images, sobel_x)
    #grads_dict['pool5_sobel_y'] = tf.gradients(pool5, images, sobel_y)
    #grads_dict['pool5_min_sobel_x'] = tf.gradients(pool5, images, (pool5-sobel_x))
    #grads_dict['pool5_min_sobel_y'] = tf.gradients(pool5, images, (pool5-sobel_y))

    # trials/1
    #grads_dict['conv1_1'] = tf.gradients(conv1_1, images, conv1_1)
    #print(grads_dict['conv1_1'][0].get_shape())
    #grads_dict['2_conv1_1'] = tf.gradients(
    #        #grads_dict['conv1_1'], images, images)
    #        grads_dict['conv1_1'], images,grads_dict['conv1_1'])
    
    #grads_dict['conv1_2'] = tf.gradients(conv1_2, images, conv1_2)
    #grads_dict['2_conv1_2'] = tf.gradients(
    #        grads_dict['conv1_2'], images,grads_dict['conv1_2'])
    #grads_dict['pool1'] = tf.gradients(pool1, images, pool1)
    #grads_dict['2_pool1'] = tf.gradients(
    #        grads_dict['pool1'], images, grads_dict['pool1'])
    #grads_dict['pool2'] = tf.gradients(pool2, images, pool2)
    #grads_dict['pool3'] = tf.gradients(pool3, images, pool3)
    #grads_dict['pool4'] = tf.gradients(pool4, images, pool4)
    #grads_dict['pool5'] = tf.gradients(pool5, images, pool5)

    # 
    #grads_dict['pool1_conv'] = tf.gradients(pool1, conv1_2)
    #grads_dict['pool2_conv'] = tf.gradients(pool2, conv2_2)
    #grads_dict['pool3_conv'] = tf.gradients(pool3, conv3_3)
    #grads_dict['pool4_conv'] = tf.gradients(pool4, conv4_3)
    #grads_dict['pool5_conv'] = tf.gradients(pool5, conv5_3)

    # trials/1
    # yayayaya
    #grads_dict['pool1_conv'] = tf.gradients(pool1, conv1_2, pool1)
    #grads_dict['pool2_conv'] = tf.gradients(pool2, conv2_2, pool2)
    #grads_dict['pool3_conv'] = tf.gradients(pool3, conv3_3, pool3)
    #grads_dict['pool4_conv'] = tf.gradients(pool4, conv4_3, pool4)
    #grads_dict['pool5_conv'] = tf.gradients(pool5, conv5_3, pool5)

    # grad prop from 5
    #grads_dict['pool5'] = tf.gradients(pool5, conv5_3, pool5)
    #grads_dict['pool4'] = tf.gradients(pool4, conv4_3, grads_dict['pool5'])
    #grads_dict['pool3'] = tf.gradients(pool3, conv3_3, grads_dict['pool4'])
    #grads_dict['pool2'] = tf.gradients(pool2, conv2_2, grads_dict['pool3'])
    ##grads_dict['pool1'] = tf.gradients(pool1, conv1_2, grads_dict['pool2'])
    #grads_dict['pool1'] = tf.gradients(pool1, images, grads_dict['pool2'])

    # fail if you prop it not from 5
    #grads_dict['pool5'] = tf.gradients(pool5, conv5_3, pool5)
    #grads_dict['pool4'] = tf.gradients(pool4, conv4_3, pool4)
    #grads_dict['pool3'] = tf.gradients(pool3, conv3_3, grads_dict['pool4'])
    #grads_dict['pool2'] = tf.gradients(pool2, conv2_2, grads_dict['pool3'])
    ##grads_dict['pool1'] = tf.gradients(pool1, conv1_2, grads_dict['pool2'])
    #grads_dict['pool1'] = tf.gradients(pool1, images, grads_dict['pool2'])

    #grads_dict['pool1_conv'] = tf.gradients(pool1, conv1_1, pool1)
    #grads_dict['pool2_conv'] = tf.gradients(pool2, conv2_1, pool2)
    #grads_dict['pool3_conv'] = tf.gradients(pool3, conv3_2, pool3)
    #grads_dict['pool4_conv'] = tf.gradients(pool4, conv4_2, pool4)
    #grads_dict['pool5_conv'] = tf.gradients(pool5, conv5_2, pool5)


    #grads_dict['pool1_conv'] = tf.gradients(pool1, images, pool1)
    #grads_dict['pool2_conv'] = tf.gradients(pool2, pool1, pool2)
    #grads_dict['pool3_conv'] = tf.gradients(pool3, pool2, pool3)
    #grads_dict['pool4_conv'] = tf.gradients(pool4, pool3, pool4)
    #grads_dict['pool5_conv'] = tf.gradients(pool5, pool4, pool5)

    # trials/2
    #grads_dict['pool1'] = tf.gradients(pool1, conv1_2, pool1)
    #grads_dict['pool2'] = tf.gradients(pool2, conv1_2, pool2)
    #grads_dict['pool3'] = tf.gradients(pool3, conv1_2, pool3)
    #grads_dict['pool4'] = tf.gradients(pool4, conv1_2, pool4)
    #grads_dict['pool5'] = tf.gradients(pool5, conv1_2, pool5)
    
    # trials/3 
    #grads_dict['pool1'] = tf.gradients(pool1, conv1_2)
    #grads_dict['pool2'] = tf.gradients(pool2, conv1_2)
    #grads_dict['pool3'] = tf.gradients(pool3, conv1_2)
    #grads_dict['pool4'] = tf.gradients(pool4, conv1_2)
    #grads_dict['pool5'] = tf.gradients(pool5, conv1_2)

    #pool4 = tf.nn.l2_normalize(pool4, dim=3, epsilon=1e-12)
    
    #feat['pool1'] = pool1
    #feat['pool2'] = pool2
    #feat['pool3'] = pool3
    #feat['pool3'] = tf.nn.l2_normalize(feat['pool3'], dim=3, epsilon=1e-12)
    #feat['conv3_3'] = conv3_3
    #feat['pool4'] = pool4
    #feat['pool4'] = tf.nn.l2_normalize(feat['pool4'], dim=3, epsilon=1e-12)
    #feat['pool5'] = pool5

    # grad as feature trial 1
    #feat['pool5'] = tf.gradients(pool5, pool4, pool5)
    #feat['pool4'] = tf.gradients(pool4, pool3, pool4)
    #feat['pool4'] = tf.nn.l2_normalize(feat['pool4'], dim=3, epsilon=1e-12)
    #feat['pool3'] = tf.gradients(pool5, pool3, pool5)
    #feat['pool3'] = tf.gradients(pool3, pool2, pool3)

    return feat, grads_dict



def model_grad_deconv(images, is_training=False, reuse=False):
    """ 
    Feature backpropagation following the deconv net rules i.e. backpropagate
    only the positive values of the local gradients i.e. gradient of a feature
    map with its predecessor. 
    More formally, as written in guided
    backprop paper: R^l_i = (R^{l+1}_i > 0)*R^{l+1}_i
    Obs: Visually the same as previous backprop so I am wondering if the
    implementation of the grad of the relu in Tf does not already does this.
    Args:
      images: [batch)size, H, W, C]
      is_training: True if traning mode (for batchnorm)
    """
    
    if not reuse:
        print('inference::input', images.get_shape())
    bn = False
    grads_dict = {}
    argmax = {}
    feat = {}
 
    with tf.variable_scope('conv1_1', reuse=reuse) as scope: #1
        conv1_1 = tf.layers.conv2d(inputs=images, filters=64, kernel_size=(3,3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv1_1 = tf.contrib.layers.batch_norm(conv1_1, fused=True, decay=0.9, is_training=is_training)
        conv1_1 = tf.nn.relu(conv1_1)
        if not reuse:
            print('conv1_1', conv1_1.get_shape())
        ##grads_dict['conv1_1'] = tf.gradients(conv1_1, images, conv1_1)
  
    with tf.variable_scope('conv1_2', reuse=reuse) as scope: #2
        conv1_2 = tf.layers.conv2d(inputs=conv1_1, filters=64, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv1_2 = tf.contrib.layers.batch_norm(conv1_2, fused=True, decay=0.9, is_training=is_training)
        conv1_2 = tf.nn.relu(conv1_2)
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool1
        #pool1,argmax['pool1'] = tf.nn.max_pool_with_argmax(conv1_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool1
        if not reuse:
            print('pool1', pool1.get_shape())
        #grads_dict['conv1_2'] = tf.gradients(conv1_2, images, conv1_2)
        #grads_dict['pool1'] = tf.gradients(pool1, conv1_2, pool1)
        #feat['pool1'] = pool1
  
    with tf.variable_scope('conv2_1', reuse=reuse) as scope:#3
        conv2_1 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv2_1 = tf.contrib.layers.batch_norm(conv2_1, fused=True, decay=0.9, is_training=is_training)
        conv2_1 = tf.nn.relu(conv2_1)
        if not reuse:
            print('conv2_1', conv2_1.get_shape())
        #grads_dict['conv2_1'] = tf.gradients(conv2_1, images, conv2_1)

    with tf.variable_scope('conv2_2', reuse=reuse) as scope:#4
        conv2_2 = tf.layers.conv2d(inputs=conv2_1, filters=128, kernel_size=(3, 3),
            padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv2_2 = tf.contrib.layers.batch_norm(conv2_2, fused=True, decay=0.9, is_training=is_training)
        conv2_2 = tf.nn.relu(conv2_2)
        pool2 = tf.nn.max_pool(conv2_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool2
        #pool2, argmax['pool2'] = tf.nn.max_pool_with_argmax(conv2_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool2
        if not reuse:
            print('pool2', pool2.get_shape())
        #grads_dict['conv2_2'] = tf.gradients(conv2_2, images, conv2_2)
        #grads_dict['pool2'] = tf.gradients(pool2, images, pool2)
  
    with tf.variable_scope('conv3_1', reuse=reuse) as scope:#5
        conv3_1 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv3_1 = tf.contrib.layers.batch_norm(conv3_1, fused=True, decay=0.9, is_training=is_training)
        conv3_1 = tf.nn.relu(conv3_1)
        if not reuse:
            print('conv3_1', conv3_1.get_shape())
        #grads_dict['conv3_1'] = tf.gradients(conv3_1, images, conv3_1)

    with tf.variable_scope('conv3_2', reuse=reuse) as scope:#6
        conv3_2 = tf.layers.conv2d(inputs=conv3_1, filters=256, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv3_2 = tf.contrib.layers.batch_norm(conv3_2, fused=True, decay=0.9, is_training=is_training)
        conv3_2 = tf.nn.relu(conv3_2)
        if not reuse:
            print('conv3_2', conv3_2.get_shape())
        #grads_dict['conv3_2'] = tf.gradients(conv3_2, images, conv3_2)

    with tf.variable_scope('conv3_3', reuse=reuse) as scope:#7
        conv3_3 = tf.layers.conv2d(inputs=conv3_2, filters=256, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv3_3 = tf.contrib.layers.batch_norm(conv3_3, fused=True, decay=0.9, is_training=is_training)
        conv3_3 = tf.nn.relu(conv3_3)
        pool3 = tf.nn.max_pool(conv3_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool3
        #pool3, argmax['pool3'] = tf.nn.max_pool_with_argmax(conv3_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool3
        if not reuse:
            print('pool3', pool3.get_shape())
        #grads_dict['conv3_3'] = tf.gradients(conv3_3, images, conv3_3)
        #grads_dict['pool3'] = tf.gradients(pool3, images, pool3)

    with tf.variable_scope('conv4_1', reuse=reuse) as scope:# 8
        conv4_1 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv4_1 = tf.contrib.layers.batch_norm(conv4_1, fused=True, decay=0.9, is_training=is_training)
        conv4_1 = tf.nn.relu(conv4_1)
        if not reuse:
            print('conv4_1', conv4_1.get_shape())
        #grads_dict['conv4_1'] = tf.gradients(conv4_1, images, conv4_1)

    with tf.variable_scope('conv4_2', reuse=reuse) as scope:#9
        conv4_2 = tf.layers.conv2d(inputs=conv4_1, filters=512, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv4_2 = tf.contrib.layers.batch_norm(conv4_2, fused=True, decay=0.9, is_training=is_training)
        conv4_2 = tf.nn.relu(conv4_2)
        if not reuse:
            print('conv4_2', conv4_2.get_shape())
        #grads_dict['conv4_2'] = tf.gradients(conv4_2, images, conv4_2)

    with tf.variable_scope('conv4_3', reuse=reuse) as scope:#10
        conv4_3 = tf.layers.conv2d(inputs=conv4_2, filters=512, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv4_3 = tf.contrib.layers.batch_norm(conv4_3, fused=True, decay=0.9, is_training=is_training)
        conv4_3 = tf.nn.relu(conv4_3)
        pool4 = tf.nn.max_pool(conv4_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool4
        #pool4, argmax['pool4'] = tf.nn.max_pool_with_argmax(conv4_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool4
        if not reuse:
            print('pool4', pool4.get_shape())
        #grads_dict['conv4_3'] = tf.gradients(conv4_3, images, conv4_3)
        #grads_dict['pool4'] = tf.gradients(pool4, images, pool4)

    with tf.variable_scope('conv5_1', reuse=reuse) as scope:#11
        conv5_1 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv5_1 = tf.contrib.layers.batch_norm(conv5_1, fused=True, decay=0.9, is_training=is_training)
        conv5_1 = tf.nn.relu(conv5_1)
        if not reuse:
            print('conv5_1', conv5_1.get_shape())
        #grads_dict['conv5_1'] = tf.gradients(conv5_1, images, conv5_1)
    
    with tf.variable_scope('conv5_2', reuse=reuse) as scope:#12
        conv5_2 = tf.layers.conv2d(inputs=conv5_1, filters=512, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv5_2 = tf.contrib.layers.batch_norm(conv5_2, fused=True, decay=0.9, is_training=is_training)
        conv5_2 = tf.nn.relu(conv5_2)
        if not reuse:
            print('conv5_2', conv5_2.get_shape())
        #grads_dict['conv5_2'] = tf.gradients(conv5_2, images, conv5_2)

    with tf.variable_scope('conv5_3', reuse=reuse) as scope:#13
        conv5_3 = tf.layers.conv2d(inputs=conv5_2, filters=512, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv5_3 = tf.contrib.layers.batch_norm(conv5_3, fused=True, decay=0.9, is_training=is_training)
        conv5_3 = tf.nn.relu(conv5_3)
        pool5 = tf.nn.max_pool(conv5_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool5
        #pool5, argmax['pool5'] = tf.nn.max_pool_with_argmax(conv5_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool5
        if not reuse:
            print('pool5', pool5.get_shape())
        #grads_dict['conv5_3'] = tf.gradients(conv5_3, images, conv5_3)
        #grads_dict['pool5'] = tf.gradients(pool5, images, pool5)
    
    feat['pool3'] = pool3
    feat['pool4'] = pool4
    feat['pool5'] = pool5
    #grads_dict['pool1_raw'] = tf.gradients(pool1, images, pool1)
    #grads_dict['pool2_raw'] = tf.gradients(pool2, images, pool2)
    #grads_dict['pool3_raw'] = tf.gradients(pool3, images, pool3)
    #grads_dict['pool4_raw'] = tf.gradients(pool4, images, pool4)
    #grads_dict['pool5_raw'] = tf.gradients(pool5, images, pool5)

    DECONV = (0==1)
    # random idea of mine to take the absolute value of features
    # before backpropagating them
    ABS = (1==1) 
    
    GUIDED_BACKPROP = (0==1)

    # same as backprop so maybe tf implements relu backprop using deconv
    # I ay have miscoded it because I don't undestand why it is the same
    if GUIDED_BACKPROP:
        # grad for 1
        grads_dict['pool1'] = tf.gradients(pool1, images, pool1)
        grads_dict['pool1'] = grads_dict['pool1']*(grads_dict['pool1']>0)
        #grads_dict['pool1'] = grads_dict['pool1']*(pool1>0)

        # grad for 2
        grads_dict['pool2'] = tf.gradients(pool2, pool1, pool2)
        grads_dict['pool2'] = grads_dict['pool2']*(grads_dict['pool2']>0)
        print('grads_dict[pool2].shape', grads_dict['pool2'][0].get_shape())
        grads_dict['pool2'] = grads_dict['pool2'][0]*tf.cast((pool1>0), tf.float32)
        grads_dict['pool2'] = tf.gradients(pool1, images, grads_dict['pool2'])
        grads_dict['pool2'] = grads_dict['pool2']*(grads_dict['pool2']>0)
        #grads_dict['pool2'] = grads_dict['pool2']*(pool2>0)
 
        # grad for 3
        grads_dict['pool3'] = tf.gradients(pool3, pool2, pool3)
        grads_dict['pool3'] = grads_dict['pool3']*(grads_dict['pool3']>0)
        grads_dict['pool3'] = grads_dict['pool3'][0]*tf.cast(pool2>0, tf.float32)
        #print('grads_dict[pool3].shape', grads_dict['pool3'][0].get_shape())
        grads_dict['pool3'] = tf.gradients(pool2, pool1, grads_dict['pool3'])
        grads_dict['pool3'] = grads_dict['pool3']*(grads_dict['pool3']>0)
        grads_dict['pool3'] = grads_dict['pool3'][0]*tf.cast(pool1>0, tf.float32)
        grads_dict['pool3'] = tf.gradients(pool1, images, grads_dict['pool3'])
        grads_dict['pool3'] = grads_dict['pool3']*(grads_dict['pool3']>0)
        #grads_dict['pool3'] = grads_dict['pool3']*(pool3>0)
 
        # grad for 4
        grads_dict['pool4'] = tf.gradients(pool4, pool3, pool4)
        grads_dict['pool4'] = grads_dict['pool4']*(grads_dict['pool4']>0)
        grads_dict['pool4'] = grads_dict['pool4'][0]*tf.cast(pool3>0, tf.float32)
        grads_dict['pool4'] = tf.gradients(pool3,  pool2, grads_dict['pool4'])
        grads_dict['pool4'] = grads_dict['pool4']*(grads_dict['pool4']>0)
        grads_dict['pool4'] = grads_dict['pool4'][0]*tf.cast(pool2>0, tf.float32)
        grads_dict['pool4'] = tf.gradients(pool2, pool1, grads_dict['pool4'])
        grads_dict['pool4'] = grads_dict['pool4']*(grads_dict['pool4']>0)
        grads_dict['pool4'] = grads_dict['pool4'][0]*tf.cast(pool1>0, tf.float32)
        grads_dict['pool4'] = tf.gradients(pool1, images, grads_dict['pool4'])
        grads_dict['pool4'] = grads_dict['pool4']*(grads_dict['pool4']>0)
        #grads_dict['pool4'] = grads_dict['pool4']*(pool4>0)
 
        # grad for 5
        grads_dict['pool5'] = tf.gradients(pool5, pool4, pool5)
        grads_dict['pool5'] = grads_dict['pool5']*(grads_dict['pool5']>0)
        grads_dict['pool5'] = grads_dict['pool5'][0]*tf.cast(pool4>0, tf.float32)
        grads_dict['pool5'] = tf.gradients(pool4,  pool3, grads_dict['pool5'])
        grads_dict['pool5'] = grads_dict['pool5']*(grads_dict['pool5']>0)
        grads_dict['pool5'] = grads_dict['pool5'][0]*tf.cast(pool3>0, tf.float32)
        grads_dict['pool5'] = tf.gradients(pool3, pool2, grads_dict['pool5'])
        grads_dict['pool5'] = grads_dict['pool5']*(grads_dict['pool5']>0)
        grads_dict['pool5'] = grads_dict['pool5'][0]*tf.cast(pool2>0, tf.float32)
        grads_dict['pool5'] = tf.gradients(pool2, pool1, grads_dict['pool5'])
        grads_dict['pool5'] = grads_dict['pool5']*(grads_dict['pool5']>0)
        grads_dict['pool5'] = grads_dict['pool5'][0]*tf.cast(pool1>0, tf.float32)
        grads_dict['pool5'] = tf.gradients(pool1, images, grads_dict['pool5'])
        grads_dict['pool5'] = grads_dict['pool5']*(grads_dict['pool5']>0)
        #grads_dict['pool5'] = grads_dict['pool5']*(pool5>0)

    # gradmaps are less noisy. Test your stuff on this gradient. 
    # yes it means more tests ... :(
    if ABS:
        # grad for 1
        grads_dict['pool1'] = tf.gradients(pool1, images, pool1)
        #grads_dict['pool1'] = tf.abs(grads_dict['pool1'])
        #print(grads_dict['pool1'][0].get_shape())
        #raw_input('wait')

        # grad for 2
        grads_dict['pool2'] = tf.gradients(pool2, pool1, pool2)
        grads_dict['pool2'] = tf.nn.relu(grads_dict['pool2'][0]) + tf.nn.relu(-grads_dict['pool2'][0])
        grads_dict['pool2'] = tf.gradients(pool1, images, grads_dict['pool2'])
        #grads_dict['pool2'] = tf.abs(grads_dict['pool2'])
 
        ## grad for 3
        grads_dict['pool3'] = tf.gradients(pool3, pool2, pool3)
        grads_dict['pool3'] = tf.nn.relu(grads_dict['pool3'][0]) + tf.nn.relu(-grads_dict['pool3'][0])
        grads_dict['pool3'] = tf.gradients(pool2, pool1, grads_dict['pool3'])
        grads_dict['pool3'] = tf.nn.relu(grads_dict['pool3'][0]) + tf.nn.relu(-grads_dict['pool3'][0])
        grads_dict['pool3'] = tf.gradients(pool1, images, grads_dict['pool3'])
        #grads_dict['pool3'] = tf.nn.relu(grads_dict['pool3'][0]) + tf.nn.relu(-grads_dict['pool3'][0])
        #print(grads_dict['pool1'][0].get_shape())
        #raw_input('wait')

        ## grad for 4
        grads_dict['pool4'] = tf.gradients(pool4, pool4, pool4)
        grads_dict['pool4'] = tf.nn.relu(grads_dict['pool4'][0]) + tf.nn.relu(-grads_dict['pool4'][0])
        grads_dict['pool4'] = tf.gradients(pool4,  pool2, grads_dict['pool4'])
        grads_dict['pool4'] = tf.nn.relu(grads_dict['pool4'][0]) + tf.nn.relu(-grads_dict['pool4'][0])
        grads_dict['pool4'] = tf.gradients(pool2, pool1, grads_dict['pool4'])
        grads_dict['pool4'] = tf.nn.relu(grads_dict['pool4'][0]) + tf.nn.relu(-grads_dict['pool4'][0])
        grads_dict['pool4'] = tf.gradients(pool1, images, grads_dict['pool4'])
 
        ## grad for 5
        grads_dict['pool5'] = tf.gradients(pool5, pool5, pool5)
        grads_dict['pool5'] = tf.nn.relu(grads_dict['pool5'][0]) + tf.nn.relu(-grads_dict['pool5'][0])
        grads_dict['pool5'] = tf.gradients(pool5,  pool3, grads_dict['pool5'])
        grads_dict['pool5'] = tf.nn.relu(grads_dict['pool5'][0]) + tf.nn.relu(-grads_dict['pool5'][0])
        grads_dict['pool5'] = tf.gradients(pool3, pool2, grads_dict['pool5'])
        grads_dict['pool5'] = tf.nn.relu(grads_dict['pool5'][0]) + tf.nn.relu(-grads_dict['pool5'][0])
        grads_dict['pool5'] = tf.gradients(pool2, pool1, grads_dict['pool5'])
        grads_dict['pool5'] = tf.nn.relu(grads_dict['pool5'][0]) + tf.nn.relu(-grads_dict['pool5'][0])
        grads_dict['pool5'] = tf.gradients(pool1, images, grads_dict['pool5'])
    
    # same as backprop so maybe tf implements relu backprop using deconv
    if DECONV:
        grads_dict['pool1_raw'] = tf.gradients(pool1, images, pool1)
        grads_dict['pool2_raw'] = tf.gradients(pool2, images, pool2)
        grads_dict['pool3_raw'] = tf.gradients(pool3, images, pool3)
        grads_dict['pool4_raw'] = tf.gradients(pool4, images, pool4)
        grads_dict['pool5_raw'] = tf.gradients(pool5, images, pool5)

        # grad for 1
        grads_dict['pool1'] = tf.gradients(pool1, images, pool1)
        grads_dict['pool1'] = grads_dict['pool1']*(grads_dict['pool1']>0)

        # grad for 2
        grads_dict['pool2'] = tf.gradients(pool2, pool1, pool2)
        grads_dict['pool2'] = grads_dict['pool2']*(grads_dict['pool2']>0)
        grads_dict['pool2'] = tf.gradients(pool1, images, grads_dict['pool2'])
        grads_dict['pool2'] = grads_dict['pool2']*(grads_dict['pool2']>0)
 
        # grad for 3
        grads_dict['pool3'] = tf.gradients(pool3, pool2, pool3)
        grads_dict['pool3'] = grads_dict['pool3']*(grads_dict['pool3']>0)
        grads_dict['pool3'] = tf.gradients(pool2, pool1, grads_dict['pool3'])
        grads_dict['pool3'] = grads_dict['pool3']*(grads_dict['pool3']>0)
        grads_dict['pool3'] = tf.gradients(pool1, images, grads_dict['pool3'])
        grads_dict['pool3'] = grads_dict['pool3']*(grads_dict['pool3']>0)
 
        # grad for 4
        grads_dict['pool4'] = tf.gradients(pool4, pool3, pool4)
        grads_dict['pool4'] = grads_dict['pool4']*(grads_dict['pool4']>0)
        grads_dict['pool4'] = tf.gradients(pool3,  pool2, grads_dict['pool4'])
        grads_dict['pool4'] = grads_dict['pool4']*(grads_dict['pool4']>0)
        grads_dict['pool4'] = tf.gradients(pool2, pool1, grads_dict['pool4'])
        grads_dict['pool4'] = grads_dict['pool4']*(grads_dict['pool4']>0)
        grads_dict['pool4'] = tf.gradients(pool1, images, grads_dict['pool4'])
        grads_dict['pool4'] = grads_dict['pool4']*(grads_dict['pool4']>0)
 
        # grad for 5
        grads_dict['pool5'] = tf.gradients(pool5, pool4, pool5)
        grads_dict['pool5'] = grads_dict['pool5']*(grads_dict['pool5']>0)
        grads_dict['pool5'] = tf.gradients(pool4,  pool3, grads_dict['pool5'])
        grads_dict['pool5'] = grads_dict['pool5']*(grads_dict['pool5']>0)
        grads_dict['pool5'] = tf.gradients(pool3, pool2, grads_dict['pool5'])
        grads_dict['pool5'] = grads_dict['pool5']*(grads_dict['pool5']>0)
        grads_dict['pool5'] = tf.gradients(pool2, pool1, grads_dict['pool5'])
        grads_dict['pool5'] = grads_dict['pool5']*(grads_dict['pool5']>0)
        grads_dict['pool5'] = tf.gradients(pool1, images, grads_dict['pool5'])
        grads_dict['pool5'] = grads_dict['pool5']*(grads_dict['pool5']>0)

    return feat, grads_dict

def model_grad_prev(images, is_training, reuse=False):
    """ Gradient of feature map with their respect to the previous feature map
    Args:
      images: [batch)size, H, W, C]
      is_training: True if traning mode (for batchnorm)
    """
    
    if not reuse:
        print('inference::input', images.get_shape())
    bn = False
    grads_dict = {}
    argmax = {}
    feat = {}
 
    with tf.variable_scope('conv1_1', reuse=reuse) as scope: #1
        conv1_1 = tf.layers.conv2d(inputs=images, filters=64, kernel_size=(3,3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv1_1 = tf.contrib.layers.batch_norm(conv1_1, fused=True, decay=0.9, is_training=is_training)
        conv1_1 = tf.nn.relu(conv1_1)
        if not reuse:
            print('conv1_1', conv1_1.get_shape())
  
    with tf.variable_scope('conv1_2', reuse=reuse) as scope: #2
        conv1_2 = tf.layers.conv2d(inputs=conv1_1, filters=64, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv1_2 = tf.contrib.layers.batch_norm(conv1_2, fused=True, decay=0.9, is_training=is_training)
        conv1_2 = tf.nn.relu(conv1_2)
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool1
        #pool1,argmax['pool1'] = tf.nn.max_pool_with_argmax(conv1_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool1
        if not reuse:
            print('pool1', pool1.get_shape())
        
        #feat['pool1'] = pool1
  
    with tf.variable_scope('conv2_1', reuse=reuse) as scope:#3
        conv2_1 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv2_1 = tf.contrib.layers.batch_norm(conv2_1, fused=True, decay=0.9, is_training=is_training)
        conv2_1 = tf.nn.relu(conv2_1)
        if not reuse:
            print('conv2_1', conv2_1.get_shape())

    with tf.variable_scope('conv2_2', reuse=reuse) as scope:#4
        conv2_2 = tf.layers.conv2d(inputs=conv2_1, filters=128, kernel_size=(3, 3),
            padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv2_2 = tf.contrib.layers.batch_norm(conv2_2, fused=True, decay=0.9, is_training=is_training)
        conv2_2 = tf.nn.relu(conv2_2)
        pool2 = tf.nn.max_pool(conv2_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool2
        #pool2, argmax['pool2'] = tf.nn.max_pool_with_argmax(conv2_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool2
        if not reuse:
            print('pool2', pool2.get_shape())

  
    with tf.variable_scope('conv3_1', reuse=reuse) as scope:#5
        conv3_1 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv3_1 = tf.contrib.layers.batch_norm(conv3_1, fused=True, decay=0.9, is_training=is_training)
        conv3_1 = tf.nn.relu(conv3_1)
        if not reuse:
            print('conv3_1', conv3_1.get_shape())

    with tf.variable_scope('conv3_2', reuse=reuse) as scope:#6
        conv3_2 = tf.layers.conv2d(inputs=conv3_1, filters=256, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv3_2 = tf.contrib.layers.batch_norm(conv3_2, fused=True, decay=0.9, is_training=is_training)
        conv3_2 = tf.nn.relu(conv3_2)
        if not reuse:
            print('conv3_2', conv3_2.get_shape())

    with tf.variable_scope('conv3_3', reuse=reuse) as scope:#7
        conv3_3 = tf.layers.conv2d(inputs=conv3_2, filters=256, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv3_3 = tf.contrib.layers.batch_norm(conv3_3, fused=True, decay=0.9, is_training=is_training)
        conv3_3 = tf.nn.relu(conv3_3)
        pool3 = tf.nn.max_pool(conv3_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool3
        #pool3, argmax['pool3'] = tf.nn.max_pool_with_argmax(conv3_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool3
        if not reuse:
            print('pool3', pool3.get_shape())


    with tf.variable_scope('conv4_1', reuse=reuse) as scope:# 8
        conv4_1 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv4_1 = tf.contrib.layers.batch_norm(conv4_1, fused=True, decay=0.9, is_training=is_training)
        conv4_1 = tf.nn.relu(conv4_1)
        if not reuse:
            print('conv4_1', conv4_1.get_shape())

    with tf.variable_scope('conv4_2', reuse=reuse) as scope:#9
        conv4_2 = tf.layers.conv2d(inputs=conv4_1, filters=512, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv4_2 = tf.contrib.layers.batch_norm(conv4_2, fused=True, decay=0.9, is_training=is_training)
        conv4_2 = tf.nn.relu(conv4_2)
        if not reuse:
            print('conv4_2', conv4_2.get_shape())

    with tf.variable_scope('conv4_3', reuse=reuse) as scope:#10
        conv4_3 = tf.layers.conv2d(inputs=conv4_2, filters=512, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv4_3 = tf.contrib.layers.batch_norm(conv4_3, fused=True, decay=0.9, is_training=is_training)
        conv4_3 = tf.nn.relu(conv4_3)
        pool4 = tf.nn.max_pool(conv4_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool4
        #pool4, argmax['pool4'] = tf.nn.max_pool_with_argmax(conv4_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool4
        if not reuse:
            print('pool4', pool4.get_shape())


    with tf.variable_scope('conv5_1', reuse=reuse) as scope:#11
        conv5_1 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv5_1 = tf.contrib.layers.batch_norm(conv5_1, fused=True, decay=0.9, is_training=is_training)
        conv5_1 = tf.nn.relu(conv5_1)
        if not reuse:
            print('conv5_1', conv5_1.get_shape())
    
    with tf.variable_scope('conv5_2', reuse=reuse) as scope:#12
        conv5_2 = tf.layers.conv2d(inputs=conv5_1, filters=512, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv5_2 = tf.contrib.layers.batch_norm(conv5_2, fused=True, decay=0.9, is_training=is_training)
        conv5_2 = tf.nn.relu(conv5_2)
        if not reuse:
            print('conv5_2', conv5_2.get_shape())

    with tf.variable_scope('conv5_3', reuse=reuse) as scope:#13
        conv5_3 = tf.layers.conv2d(inputs=conv5_2, filters=512, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv5_3 = tf.contrib.layers.batch_norm(conv5_3, fused=True, decay=0.9, is_training=is_training)
        conv5_3 = tf.nn.relu(conv5_3)
        pool5 = tf.nn.max_pool(conv5_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool5
        #pool5, argmax['pool5'] = tf.nn.max_pool_with_argmax(conv5_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool5
        if not reuse:
            print('pool5', pool5.get_shape())

    feat['pool1'] = pool1
    feat['pool2'] = pool2
    feat['pool3'] = pool3
    feat['pool4'] = pool4
    feat['pool5'] = pool5

    grads_dict['conv1_1'] = tf.gradients(conv1_1, images, conv1_1)
    grads_dict['conv1_2'] = tf.gradients(conv1_2, conv1_1, conv1_2)
    grads_dict['pool1'] = tf.gradients(pool1, conv1_2, pool1)
    grads_dict['conv2_1'] = tf.gradients(conv2_1, pool1, conv2_1)
    grads_dict['conv2_2'] = tf.gradients(conv2_2, conv2_1, conv2_2)
    grads_dict['pool2'] = tf.gradients(pool2, conv2_2, pool2)
    grads_dict['conv3_1'] = tf.gradients(conv3_1, pool2, conv3_1)
    grads_dict['conv3_2'] = tf.gradients(conv3_2, conv3_1, conv3_2)
    grads_dict['conv3_3'] = tf.gradients(conv3_3, conv3_2, conv3_3)
    grads_dict['pool3'] = tf.gradients(pool3, conv3_3, pool3)
    grads_dict['conv4_1'] = tf.gradients(conv4_1, pool3, conv4_1)
    grads_dict['conv4_2'] = tf.gradients(conv4_2, conv4_1, conv4_2)
    grads_dict['conv4_3'] = tf.gradients(conv4_3, conv4_2, conv4_3)
    grads_dict['pool4'] = tf.gradients(pool4, conv4_3, pool4)
    grads_dict['conv5_1'] = tf.gradients(conv5_1, pool4, conv5_1)
    grads_dict['conv5_2'] = tf.gradients(conv5_2, conv5_1, conv5_2)
    grads_dict['conv5_3'] = tf.gradients(conv5_3, conv5_2, conv5_3)
    grads_dict['pool5'] = tf.gradients(pool5, conv5_2, pool5)

    return feat, grads_dict


def model_nopool(images, is_training, reuse=False):
    """ Network model
    Args:
      images: Images returned from distorted_inputs() or inputs().
    Returns:
      Logits.
    """
    
    print('inference::input', images.get_shape())
    bn = False
    feat, grads_dict = {},{}
 
    with tf.variable_scope('conv1_1', reuse=reuse) as scope: #1
        conv1_1 = tf.layers.conv2d(inputs=images, filters=64, kernel_size=(3,3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv1_1 = tf.contrib.layers.batch_norm(conv1_1, fused=True, decay=0.9, is_training=is_training)
        conv1_1 = tf.nn.relu(conv1_1)
        print('conv1_1', conv1_1.get_shape())
  
    with tf.variable_scope('conv1_2', reuse=reuse) as scope: #2
        conv1_2 = tf.layers.conv2d(inputs=conv1_1, filters=64, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv1_2 = tf.contrib.layers.batch_norm(conv1_2, fused=True, decay=0.9, is_training=is_training)
        pool1 = tf.nn.relu(conv1_2)
        #conv1_2 = tf.nn.relu(conv1_2)
        #pool1 = tf.nn.max_pool(conv1_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool1
        #feat['pool1'] = pool1
        print('pool1', pool1.get_shape())
  
    with tf.variable_scope('conv2_1', reuse=reuse) as scope:#3
        conv2_1 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv2_1 = tf.contrib.layers.batch_norm(conv2_1, fused=True, decay=0.9, is_training=is_training)
        conv2_1 = tf.nn.relu(conv2_1)
        print('conv2_1', conv2_1.get_shape())

    with tf.variable_scope('conv2_2', reuse=reuse) as scope:#4
        conv2_2 = tf.layers.conv2d(inputs=conv2_1, filters=128, kernel_size=(3, 3),
            padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv2_2 = tf.contrib.layers.batch_norm(conv2_2, fused=True, decay=0.9, is_training=is_training)
        #conv2_2 = tf.nn.relu(conv2_2)
        pool2 = tf.nn.relu(conv2_2)
        #pool2 = tf.nn.max_pool(conv2_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool2
        #feat['pool2'] = pool2
        print('pool2', pool2.get_shape())
  
    with tf.variable_scope('conv3_1', reuse=reuse) as scope:#5
        conv3_1 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv3_1 = tf.contrib.layers.batch_norm(conv3_1, fused=True, decay=0.9, is_training=is_training)
        conv3_1 = tf.nn.relu(conv3_1)
        print('conv3_1', conv3_1.get_shape())

    with tf.variable_scope('conv3_2', reuse=reuse) as scope:#6
        conv3_2 = tf.layers.conv2d(inputs=conv3_1, filters=256, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv3_2 = tf.contrib.layers.batch_norm(conv3_2, fused=True, decay=0.9, is_training=is_training)
        conv3_2 = tf.nn.relu(conv3_2)
        print('conv3_2', conv3_2.get_shape())

    with tf.variable_scope('conv3_3', reuse=reuse) as scope:#7
        conv3_3 = tf.layers.conv2d(inputs=conv3_2, filters=256, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv3_3 = tf.contrib.layers.batch_norm(conv3_3, fused=True, decay=0.9, is_training=is_training)
        #conv3_3= tf.nn.relu(conv3_3)
        pool3 = tf.nn.relu(conv3_3)
        #pool3 = tf.nn.max_pool(conv3_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool3
        #pool3 = tf.nn.l2_normalize(pool3, dim=3, epsilon=1e-12)
        feat['pool3'] = pool3
        print('pool3', pool3.get_shape())

    with tf.variable_scope('conv4_1', reuse=reuse) as scope:# 8
        conv4_1 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv4_1 = tf.contrib.layers.batch_norm(conv4_1, fused=True, decay=0.9, is_training=is_training)
        conv4_1 = tf.nn.relu(conv4_1)
        print('conv4_1', conv4_1.get_shape())

    with tf.variable_scope('conv4_2', reuse=reuse) as scope:#9
        conv4_2 = tf.layers.conv2d(inputs=conv4_1, filters=512, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv4_2 = tf.contrib.layers.batch_norm(conv4_2, fused=True, decay=0.9, is_training=is_training)
        conv4_2 = tf.nn.relu(conv4_2)
        print('conv4_2', conv4_2.get_shape())

    with tf.variable_scope('conv4_3', reuse=reuse) as scope:#10
        conv4_3 = tf.layers.conv2d(inputs=conv4_2, filters=512, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv4_3 = tf.contrib.layers.batch_norm(conv4_3, fused=True, decay=0.9, is_training=is_training)
        pool4 = tf.nn.relu(conv4_3)
        #conv4_3 = tf.nn.l2_normalize(conv4_3, dim=3, epsilon=1e-12)
        #feat['conv4_3'] = conv4_3
        #pool4 = tf.nn.max_pool(conv4_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool4
        #pool4 = tf.nn.l2_normalize(pool4, dim=3, epsilon=1e-12)
        #feat['pool4'] = pool4
        print('pool4', pool4.get_shape())

    #with tf.variable_scope('conv5_1', reuse=reuse) as scope:#11
    #    conv5_1 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=(3, 3),
    #            padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
    #    if bn:
    #        conv5_1 = tf.contrib.layers.batch_norm(conv5_1, fused=True, decay=0.9, is_training=is_training)
    #    conv5_1 = tf.nn.relu(conv5_1)
    #    print('conv5_1', conv5_1.get_shape())
    #
    #with tf.variable_scope('conv5_2', reuse=reuse) as scope:#12
    #    conv5_2 = tf.layers.conv2d(inputs=conv5_1, filters=512, kernel_size=(3, 3),
    #            padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
    #    if bn:
    #        conv5_2 = tf.contrib.layers.batch_norm(conv5_2, fused=True, decay=0.9, is_training=is_training)
    #    conv5_2 = tf.nn.relu(conv5_2)
    #    print('conv5_2', conv5_2.get_shape())

    #with tf.variable_scope('conv5_3', reuse=reuse) as scope:#13
    #    conv5_3 = tf.layers.conv2d(inputs=conv5_2, filters=512, kernel_size=(3, 3),
    #            padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
    #    if bn:
    #        conv5_3 = tf.contrib.layers.batch_norm(conv5_3, fused=True, decay=0.9, is_training=is_training)
    #    pool5 = tf.nn.relu(conv5_3)
    #    #pool5 = tf.nn.max_pool(conv5_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool5
    #    #pool5 = tf.nn.l2_normalize(pool5, dim=3, epsilon=1e-12)
    #    print('pool5', pool5.get_shape())
    
    # yayayaya
    #feat['pool1'] = tf.gradients(pool1, conv1_2, pool1)
    #feat['pool2'] = tf.gradients(pool2, conv2_2, pool2)
    #feat['pool3'] = tf.gradients(pool3, conv3_3, pool3)
    #feat['pool4'] = tf.gradients(pool4, conv4_3, pool4)
    #feat['pool5'] = tf.gradients(pool5, conv5_3, pool5)
    
    # grad prop from 5
    #grads_dict['pool5'] = tf.gradients(pool5, conv5_3, pool5)
    #grads_dict['pool4'] = tf.gradients(pool4, conv4_3, grads_dict['pool5'])
    #grads_dict['pool3'] = tf.gradients(pool3, conv3_3, grads_dict['pool4'])
    #grads_dict['pool2'] = tf.gradients(pool2, conv2_2, grads_dict['pool3'])
    ##grads_dict['pool1'] = tf.gradients(pool1, conv1_2, grads_dict['pool2'])
    #grads_dict['pool1'] = tf.gradients(pool1, images, grads_dict['pool2'])

    feat['pool1'] = pool1
    feat['pool2'] = pool2
    feat['pool3'] = pool3
    feat['pool4'] = pool4
    #feat['pool5'] = pool5
    return feat, grads_dict


def model_unpool(images, idx_map, is_training, reuse=False):
    """ Network model
    Args:
      images: [batch)size, H, W, C]
      is_training: True if traning mode (for batchnorm)
    """
    
    if not reuse:
        print('inference::input', images.get_shape())
    bn = True
    grads_dict = {}
 
    with tf.variable_scope('conv1_1', reuse=reuse) as scope: #1
        conv1_1 = tf.layers.conv2d(inputs=images, filters=64, kernel_size=(3,3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv1_1 = tf.contrib.layers.batch_norm(conv1_1, fused=True, decay=0.9, is_training=is_training)
        conv1_1 = tf.nn.relu(conv1_1)
        if not reuse:
            print('conv1_1', conv1_1.get_shape())
        grads_dict['conv1_1'] = tf.gradients(conv1_1, images)
  
    with tf.variable_scope('conv1_2', reuse=reuse) as scope: #2
        conv1_2 = tf.layers.conv2d(inputs=conv1_1, filters=64, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv1_2 = tf.contrib.layers.batch_norm(conv1_2, fused=True, decay=0.9, is_training=is_training)
        conv1_2 = tf.nn.relu(conv1_2)
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool1
        if not reuse:
            print('pool1', pool1.get_shape())
        grads_dict['conv1_2'] = tf.gradients(conv1_2, images)
        grads_dict['pool1'] = tf.gradients(pool1, images)
  
    with tf.variable_scope('conv2_1', reuse=reuse) as scope:#3
        conv2_1 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv2_1 = tf.contrib.layers.batch_norm(conv2_1, fused=True, decay=0.9, is_training=is_training)
        conv2_1 = tf.nn.relu(conv2_1)
        if not reuse:
            print('conv2_1', conv2_1.get_shape())
        grads_dict['conv2_1'] = tf.gradients(conv2_1, images)

    with tf.variable_scope('conv2_2', reuse=reuse) as scope:#4
        conv2_2 = tf.layers.conv2d(inputs=conv2_1, filters=128, kernel_size=(3, 3),
            padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv2_2 = tf.contrib.layers.batch_norm(conv2_2, fused=True, decay=0.9, is_training=is_training)
        conv2_2 = tf.nn.relu(conv2_2)
        pool2 = tf.nn.max_pool(conv2_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool2
        if not reuse:
            print('pool2', pool2.get_shape())
        grads_dict['conv2_2'] = tf.gradients(conv2_2, images)
        grads_dict['pool2'] = tf.gradients(pool2, images)
  
    with tf.variable_scope('conv3_1', reuse=reuse) as scope:#5
        conv3_1 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv3_1 = tf.contrib.layers.batch_norm(conv3_1, fused=True, decay=0.9, is_training=is_training)
        conv3_1 = tf.nn.relu(conv3_1)
        if not reuse:
            print('conv3_1', conv3_1.get_shape())
        grads_dict['conv3_1'] = tf.gradients(conv3_1, images)

    with tf.variable_scope('conv3_2', reuse=reuse) as scope:#6
        conv3_2 = tf.layers.conv2d(inputs=conv3_1, filters=256, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv3_2 = tf.contrib.layers.batch_norm(conv3_2, fused=True, decay=0.9, is_training=is_training)
        conv3_2 = tf.nn.relu(conv3_2)
        if not reuse:
            print('conv3_2', conv3_2.get_shape())
        grads_dict['conv3_2'] = tf.gradients(conv3_2, images)

    with tf.variable_scope('conv3_3', reuse=reuse) as scope:#7
        conv3_3 = tf.layers.conv2d(inputs=conv3_2, filters=256, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv3_3 = tf.contrib.layers.batch_norm(conv3_3, fused=True, decay=0.9, is_training=is_training)
        conv3_3 = tf.nn.relu(conv3_3)
        pool3 = tf.nn.max_pool(conv3_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool3
        if not reuse:
            print('pool3', pool3.get_shape())
        grads_dict['conv3_3'] = tf.gradients(conv3_3, images)
        grads_dict['pool3'] = tf.gradients(pool3, images)

    with tf.variable_scope('conv4_1', reuse=reuse) as scope:# 8
        conv4_1 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv4_1 = tf.contrib.layers.batch_norm(conv4_1, fused=True, decay=0.9, is_training=is_training)
        conv4_1 = tf.nn.relu(conv4_1)
        if not reuse:
            print('conv4_1', conv4_1.get_shape())
        grads_dict['conv4_1'] = tf.gradients(conv4_1, images)

    with tf.variable_scope('conv4_2', reuse=reuse) as scope:#9
        conv4_2 = tf.layers.conv2d(inputs=conv4_1, filters=512, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv4_2 = tf.contrib.layers.batch_norm(conv4_2, fused=True, decay=0.9, is_training=is_training)
        conv4_2 = tf.nn.relu(conv4_2)
        if not reuse:
            print('conv4_2', conv4_2.get_shape())
        grads_dict['conv4_2'] = tf.gradients(conv4_2, images)

    with tf.variable_scope('conv4_3', reuse=reuse) as scope:#10
        conv4_3 = tf.layers.conv2d(inputs=conv4_2, filters=512, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv4_3 = tf.contrib.layers.batch_norm(conv4_3, fused=True, decay=0.9, is_training=is_training)
        conv4_3 = tf.nn.relu(conv4_3)
        pool4 = tf.nn.max_pool(conv4_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool4
        if not reuse:
            print('pool4', pool4.get_shape())
        grads_dict['conv4_3'] = tf.gradients(conv4_3, images)
        grads_dict['pool4'] = tf.gradients(pool4, images)

    with tf.variable_scope('conv5_1', reuse=reuse) as scope:#11
        conv5_1 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv5_1 = tf.contrib.layers.batch_norm(conv5_1, fused=True, decay=0.9, is_training=is_training)
        conv5_1 = tf.nn.relu(conv5_1)
        if not reuse:
            print('conv5_1', conv5_1.get_shape())
        grads_dict['conv5_1'] = tf.gradients(conv5_1, images)
    
    with tf.variable_scope('conv5_2', reuse=reuse) as scope:#12
        conv5_2 = tf.layers.conv2d(inputs=conv5_1, filters=512, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv5_2 = tf.contrib.layers.batch_norm(conv5_2, fused=True, decay=0.9, is_training=is_training)
        conv5_2 = tf.nn.relu(conv5_2)
        if not reuse:
            print('conv5_2', conv5_2.get_shape())
        grads_dict['conv5_2'] = tf.gradients(conv5_2, images)

    with tf.variable_scope('conv5_3', reuse=reuse) as scope:#13
        conv5_3 = tf.layers.conv2d(inputs=conv5_2, filters=512, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv5_3 = tf.contrib.layers.batch_norm(conv5_3, fused=True, decay=0.9, is_training=is_training)
        conv5_3 = tf.nn.relu(conv5_3)
        pool5 = tf.nn.max_pool(conv5_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool5
        if not reuse:
            print('pool5', pool5.get_shape())
        grads_dict['conv5_3'] = tf.gradients(conv5_3, images)
        grads_dict['pool5'] = tf.gradients(pool5, images)
    
    ########################################################################################

    #with tf.variable_scope('unpool5', reuse=reuse) as scope:#14
    #    #unpool5 = gen_nn_ops._max_pool_grad(idx_map, pool5, pool5, [1,3,3,1], [1,2,2,1],'VALID') 
    #    unpool5 = gen_nn_ops._max_pool_grad(conv5_3, pool5, pool5, [1,2,2,1], [1,2,2,1],'VALID') 
    #    print('unpool5', unpool5.get_shape())
    #    #tf.summary.image('unpool5', unpool5)
    #
    #with tf.variable_scope('unpool4', reuse=reuse) as scope:#17
    #    unpool4 = gen_nn_ops._max_pool_grad(conv4_3, pool4, unpool5, [1,2,2,1], [1,2,2,1],'VALID')
    #    conv4_1D = tf.layers.conv2d(unpool4, filters=256, kernel_size=(1,1),
    #            padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
    #    print('unpool4', unpool4.get_shape())
    #    print('conv4_1D', conv4_1D.get_shape())
    #    #tf.summary.image('unpool4', unpool4)
 
    #with tf.variable_scope('unpool3', reuse=reuse) as scope:#20
    #    #unpool3 = gen_nn_ops._max_pool_grad(conv3_3, pool3, unpool4, [1,2,2,1], [1,2,2,1],'VALID') 
    #    unpool3 = gen_nn_ops._max_pool_grad(conv3_3, pool3, conv4_1D, [1,2,2,1], [1,2,2,1],'VALID') 
    #    conv3_1D = tf.layers.conv2d(unpool3, filters=128, kernel_size=(1,1),
    #            padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
    #    print('unpool3', unpool3.get_shape())
    #    print('conv3_1D', conv3_1D.get_shape())
    #    #tf.summary.image('unpool3', unpool3)
 
    #with tf.variable_scope('unpool2', reuse=reuse) as scope:#23
    #    #unpool2 = gen_nn_ops._max_pool_grad(conv2_2, pool2, unpool3, [1,2,2,1], [1,2,2,1],'VALID') 
    #    unpool2 = gen_nn_ops._max_pool_grad(conv2_2, pool2, conv3_1D, [1,2,2,1], [1,2,2,1],'VALID') 
    #    conv2_1D = tf.layers.conv2d(unpool2, filters=64, kernel_size=(1,1),
    #            padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
    #    print('unpool2', unpool2.get_shape())
    #    print('conv2_1D', conv2_1D.get_shape())
    #    #tf.summary.image('unpool2', unpool2)

    #with tf.variable_scope('unpool1', reuse=reuse) as scope:#25
    #    #unpool1 = gen_nn_ops._max_pool_grad(conv1_2, pool1, unpool2, [1,2,2,1], [1,2,2,1],'VALID') 
    #    unpool1 = gen_nn_ops._max_pool_grad(conv1_2, pool1, conv2_1D, [1,2,2,1], [1,2,2,1],'VALID') 
    #    conv1_1D = tf.layers.conv2d(unpool1, filters=1, kernel_size=(1,1),
    #            padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
    #    print('unpool1', unpool1.get_shape())
    #    print('conv1_1D', conv1_1D.get_shape())
    #    tf.summary.image('conv1_1D', conv1_1D)
    #    print('conv1_1D.shape', conv1_1D.get_shape())

    ########################################################################################
    with tf.variable_scope('unpool5', reuse=reuse) as scope:#14
        #unpool5 = gen_nn_ops._max_pool_grad(idx_map, pool5, pool5, [1,3,3,1], [1,2,2,1],'VALID') 
        unpool5 = gen_nn_ops._max_pool_grad(conv5_3, pool5, pool5, [1,2,2,1], [1,2,2,1],'VALID') 
        if not reuse:
            print('unpool5', unpool5.get_shape())
        #tf.summary.image('unpool5', unpool5)
    
    with tf.variable_scope('unpool4', reuse=reuse) as scope:#17
        unpool4 = gen_nn_ops._max_pool_grad(conv4_3, pool4, unpool5, [1,2,2,1], [1,2,2,1],'VALID')
        #conv4_1D = tf.layers.conv2d(unpool4, filters=256, kernel_size=(1,1),
        #        padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if not reuse:
            print('unpool4', unpool4.get_shape())
        #print('conv4_1D', conv4_1D.get_shape())
        #tf.summary.image('unpool4', unpool4)
 
    with tf.variable_scope('unpool3', reuse=reuse) as scope:#20
        unpool3 = gen_nn_ops._max_pool_grad(conv3_3, pool3, unpool4, [1,2,2,1], [1,2,2,1],'VALID') 
        #unpool3 = gen_nn_ops._max_pool_grad(conv3_3, pool3, conv4_1D, [1,2,2,1], [1,2,2,1],'VALID') 
        #conv3_1D = tf.layers.conv2d(unpool3, filters=128, kernel_size=(1,1),
        #        padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if not reuse:
            print('unpool3', unpool3.get_shape())
        #print('conv3_1D', conv3_1D.get_shape())
        #tf.summary.image('unpool3', unpool3)
 
    with tf.variable_scope('unpool2', reuse=reuse) as scope:#23
        unpool2 = gen_nn_ops._max_pool_grad(conv2_2, pool2, unpool3, [1,2,2,1], [1,2,2,1],'VALID') 
        #unpool2 = gen_nn_ops._max_pool_grad(conv2_2, pool2, conv3_1D, [1,2,2,1], [1,2,2,1],'VALID') 
        #conv2_1D = tf.layers.conv2d(unpool2, filters=64, kernel_size=(1,1),
        #        padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if not reuse:
            print('unpool2', unpool2.get_shape())
        #print('conv2_1D', conv2_1D.get_shape())
        #tf.summary.image('unpool2', unpool2)

    with tf.variable_scope('unpool1', reuse=reuse) as scope:#25
        unpool1 = gen_nn_ops._max_pool_grad(conv1_2, pool1, unpool2, [1,2,2,1], [1,2,2,1],'VALID') 
        unpool1 = tf.reduce_max(unpool1, (3), keep_dims=True)
        #unpool1 = gen_nn_ops._max_pool_grad(conv1_2, pool1, conv2_1D, [1,2,2,1], [1,2,2,1],'VALID') 
        #conv1_1D = tf.layers.conv2d(unpool1, filters=1, kernel_size=(1,1),
        #        padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if not reuse:
            print('unpool1', unpool1.get_shape())
        #print('conv1_1D', conv1_1D.get_shape())
        #tf.summary.image('conv1_1D', conv1_1D)
        #print('conv1_1D.shape', conv1_1D.get_shape())

        feat = pool5
        unpool = unpool1

    return feat, unpool, grads_dict

def logits(feat1, feat2, reuse=False):

    feat = tf.concat((feat1, feat2), 1)
    print('logits_classif::feat.shape: ', feat.get_shape())
    
    with tf.variable_scope('fc9', reuse=reuse) as scope:
        fc9 = tf.nn.relu(tf.layers.dense(feat, 1000))
        print('fc9', fc9.get_shape())

    with tf.variable_scope('fc10', reuse=reuse) as scope:
        fc10 = tf.nn.relu(tf.layers.dense(fc9, 2))
        print('fc10', fc10.get_shape())

    return fc10

def vgg_logits(pool5):
    #### ReLU is killing your signal 

    with tf.variable_scope('fc6', reuse=reuse) as scope:#13
        shape = int(np.prod(pool5.get_shape()[1:]))
        #shape = pool5.get_shape().as_list()
        #dim = np.prod(shape[1:])
        #pool5_flat = tf.reshape(pool5, [-1, dim])
        #print('pool5_flat', pool5_flat.get_shape())
        in_ = tf.reshape(pool5, [-1, shape])
        print('pool5_flat', in_.get_shape())
        #fc6 = tf.nn.relu( tf.layers.dense(pool5_flat, 4096) )
        fc6 = tf.nn.relu( tf.layers.dense(in_, 4096) )
        print('fc6', fc6.get_shape())

    with tf.variable_scope('fc7', reuse=reuse) as scope:#13
        fc7 = tf.nn.relu(tf.layers.dense(fc6, 1000))
        print('fc7', fc7.get_shape())

    with tf.variable_scope('fc8', reuse) as scope:#13
        fc8 = tf.nn.relu(tf.layers.dense(fc7, 1000))
        print('fc8', fc8.get_shape())
    
    return fc8


#def loss(feat1, feat2, labels):
#    """Add Loss to all the trainable variables.
#
#    Add summary for for "Loss" and "Loss/avg".
#    Args:
#      logits: Logits from inference().
#      labels: Labels from distorted_inputs or inputs(). 2-D tensor
#              of shape [batch_size, 1]
#
#    Returns:
#      Loss tensor of type float.
#    """
#    margin = 0
#    #d_op = tf.reduce_sum(tf.square(feat1 - feat2), (1,2,3))
#    #d_op = tf.reduce_sum(tf.square(feat1 - feat2), (1))
#    #d = tf.nn.l2_loss(feat1 - feat2)# / (header.BATCH_SIZE ) #* header.IMAGE_SIZE)
#    d_op = tf.reduce_sum(tf.abs(feat1 - feat2), (1,2,3)) # paper recommends L1 to avoid local minima
#
#    print('d.shape: ', d_op.get_shape())
#    print('labels.shape: ', labels.get_shape())
#    #d_sqrt = tf.sqrt(d)
#    #loss = labels * tf.square(tf.maximum(0., margin - d_sqrt)) + (1 - labels) * d
#    #loss = (1-labels) * tf.maximum(0., margin - d) + labels * d
#    loss_b = labels * d_op + (1 - labels) * (margin - d_op)
#    print('loss_b.shape: ', loss_b.get_shape())
#    #loss = tf.reduce_sum(loss_b)
#    loss = tf.reduce_mean(loss_b)
#    tf.summary.scalar('loss', loss)
#    
#    tf.add_to_collection('losses', loss)
#    # The total loss is defined as the l2 loss plus all of the weight
#    # decay terms (L2 loss).
#    return tf.add_n(tf.get_collection('losses'), name='total_loss'), loss_b#, d_op
#    #return tf.add_n(tf.get_collection('losses'), name='total_loss')
#
#def triplet_loss(feat1, feat2, feat3, args):
#    """
#        Triplet loss
#    """
#    margin = args.margin
#    #d = tf.nn.l2_loss(feat1 - feat2)# / (header.BATCH_SIZE ) #* header.IMAGE_SIZE)
#    dp = tf.reduce_sum(tf.abs(feat1 - feat2), (1,2,3)) # P example
#    dn = tf.reduce_sum(tf.abs(feat1 - feat3), (1,2,3)) # N example
#    dp_mean = tf.reduce_mean(dp)
#    dn_mean = tf.reduce_mean(dn)
#    tf.summary.scalar('dn', dn_mean)
#    tf.summary.scalar('dp', dp_mean)
#    loss_b = tf.maximum(0.0, margin + dp - dn)
#    loss = tf.reduce_mean(loss_b) 
#    #print('dn.shape: ', dn.get_shape())
#    #print('loss_b.shape: ', loss_b.get_shape())
#    tf.summary.scalar('loss', loss)
#    
#    tf.add_to_collection('losses', loss)
#    # The total loss is defined as the l2 loss plus all of the weight
#    # decay terms (L2 loss).
#    return tf.add_n(tf.get_collection('losses'), name='total_loss'), dp, dn
#    #return tf.add_n(tf.get_collection('losses'), name='total_loss')
#def loss_classif(logits, labels):
#    """Add Loss to all the trainable variables.
#
#    Add summary for for "Loss" and "Loss/avg".
#    Args:
#      logits: Logits from inference().
#      labels: Labels from distorted_inputs or inputs(). 2-D tensor
#              of shape [batch_size, 1]
#
#    Returns:
#      Loss tensor of type float.
#    """
#    labels = tf.cast(labels, tf.int64)
#    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
#    correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
#    acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#    
#    tf.summary.scalar('loss', loss)
#    tf.summary.scalar('acc', acc)
#    tf.add_to_collection('losses', loss)
#    return tf.add_n(tf.get_collection('losses'), name='total_loss'), acc
#
#def _add_loss_summaries(total_loss):
#  """Add summaries for losses in CIFAR-10 model.
#
#  Generates moving average for all losses and associated summaries for
#  visualizing the performance of the network.
#
#  Args:
#    total_loss: Total loss from loss().
#  Returns:
#    loss_averages_op: op for generating moving averages of losses.
#  """
#  # Compute the moving average of all individual losses and the total loss.
#  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
#  losses = tf.get_collection('losses')
#  loss_averages_op = loss_averages.apply(losses + [total_loss])
#
#  # Attach a scalar summary to all individual losses and the total loss; do the
#  # same for the averaged version of the losses.
#  for l in losses + [total_loss]:
#    # Name each loss as '(raw)' and name the moving average version of the loss
#    # as the original loss name.
#    tf.summary.scalar(l.op.name +' (raw)', l)
#    tf.summary.scalar(l.op.name, loss_averages.average(l))
#
#  return loss_averages_op
#
#
#def train(total_loss, global_step, args):
#    """Train CIFAR-10 model.
#    Create an optimizer and apply to all trainable variables. Add moving
#    average for all trainable variables.
#    Args:
#      total_loss: Total loss from loss().
#      global_step: Integer Variable counting the number of training steps
#        processed.
#    Returns:
#      train_op: op for training.
#    """
#    # Generate moving averages of all losses and associated summaries.
#    loss_averages_op = _add_loss_summaries(total_loss)
#
#    #var_to_train = tf.trainable_variables()
#    #print('\nvar to train')
#    #for var in var_to_train:
#    #    print(var.op.name)
#
#    # Compute gradients.
#    with tf.control_dependencies([loss_averages_op]):
#        opt = tf.train.AdamOptimizer(args.lr, args.adam_b1, args.adam_b2, args.adam_eps)
#        
#        # TODO BN
#        update_ops =  tf.get_collection(tf.GraphKeys.UPDATE_OPS) #line for BN
#        with tf.control_dependencies(update_ops):
#            grads = opt.compute_gradients(total_loss, var_list=tf.trainable_variables())
#            apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
#    
#    # no BN
#    #   grads = opt.compute_gradients(total_loss, var_list=tf.trainable_variables())
#    #apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
#
#    # Add histograms for trainable variables.
#    for var in tf.trainable_variables():
#        tf.summary.histogram(var.op.name, var)
#
#    # Add histograms for gradients.
#    for grad, var in grads:
#        if grad is not None:
#            tf.summary.histogram(var.op.name + '/gradients', grad)
#
#    # Track the moving averages of all trainable variables.
#    variable_averages = tf.train.ExponentialMovingAverage(args.moving_average_decay, global_step)
#
#    with tf.control_dependencies([apply_gradient_op]):
#        variables_averages_op = variable_averages.apply(tf.trainable_variables())
#        #train_op = tf.no_op(name='train')
#
#    return variables_averages_op #train_op



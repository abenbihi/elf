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

def model_grad(images, is_training, reuse=False, relu=False):
    """ Network model
    Args:
      images: Images returned from distorted_inputs() or inputs().
    Returns:
      Logits.
    """
    
    print('inference::input', images.get_shape())
    feat, grads_dict = {},{}
 
    with tf.variable_scope('conv1a', reuse=reuse) as scope: #1
        conv1a = tf.layers.conv2d(inputs=images, filters=64, kernel_size=(3,3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if relu:
            conv1a = tf.nn.relu(conv1a)
        print('conv1a', conv1a.get_shape())
  
    with tf.variable_scope('conv1b', reuse=reuse) as scope: #2
        conv1b = tf.layers.conv2d(inputs=conv1a, filters=64, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if relu:
            conv1b = tf.nn.relu(conv1b)
        print('conv1b', conv1b.get_shape())
        pool1 = tf.nn.max_pool(conv1b, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool') 
        print('pool1', pool1.get_shape())
  
    with tf.variable_scope('conv2a', reuse=reuse) as scope:#3
        conv2a = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv2a = tf.nn.relu(conv2a)
        print('conv2a', conv2a.get_shape())

    with tf.variable_scope('conv2b', reuse=reuse) as scope:#4
        conv2b = tf.layers.conv2d(inputs=conv2a, filters=64, kernel_size=(3, 3),
            padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv2b = tf.nn.relu(conv2b)
        pool2 = tf.nn.max_pool(conv2b, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool') 
        print('pool2', pool2.get_shape())
  
    with tf.variable_scope('conv3a', reuse=reuse) as scope:#5
        conv3a = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv3a = tf.nn.relu(conv3a)
        print('conv3a', conv3a.get_shape())

    with tf.variable_scope('conv3b', reuse=reuse) as scope:#6
        conv3b = tf.layers.conv2d(inputs=conv3a, filters=128, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv3b = tf.nn.relu(conv3b)
        print('conv3b', conv3b.get_shape())
        pool3 = tf.nn.max_pool(conv3b, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool') 
        print('pool3', pool3.get_shape())

    with tf.variable_scope('conv4a', reuse=reuse) as scope:# 8
        conv4a = tf.layers.conv2d(inputs=pool3, filters=128, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv4a = tf.nn.relu(conv4a)
        print('conv4a', conv4a.get_shape())

    with tf.variable_scope('conv4b', reuse=reuse) as scope:#9
        conv4b = tf.layers.conv2d(inputs=conv4a, filters=128, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv4b = tf.nn.relu(conv4b)
        print('conv4b', conv4b.get_shape())

    # detector head
    with tf.variable_scope('convPa', reuse=reuse) as scope:#11
        convPa = tf.layers.conv2d(inputs=conv4b, filters=256, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        convPa = tf.nn.relu(convPa)
        print('convPa', convPa.get_shape())
    
    with tf.variable_scope('convPb', reuse=reuse) as scope:#12
        convPb = tf.layers.conv2d(inputs=convPa, filters=65, kernel_size=(1, 1),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        print('convPb', convPb.get_shape())


    # descriptor head
    with tf.variable_scope('convDa', reuse=reuse) as scope:#13
        convDa = tf.layers.conv2d(inputs=conv4b, filters=256, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        convDa = tf.nn.relu(convDa)
        print('convDa', convDa.get_shape())
    
    with tf.variable_scope('convDb', reuse=reuse) as scope:#13
        convDb = tf.layers.conv2d(inputs=convDa, filters=256, kernel_size=(1, 1),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        print('convDb', convDb.get_shape())
    
    convDb = tf.nn.l2_normalize(convDb, dim=3, epsilon=1e-12)

    # Choose your backpop
    BACKPROP = (1==1)
    GUIDED_BACKPROP = (0==1)

    # Let's go !
    if BACKPROP:
        grads_dict['conv1a'] = tf.gradients(conv1a, images, conv1a)
        grads_dict['conv1b'] = tf.gradients(conv1b, images, conv1b)
        grads_dict['pool1'] = tf.gradients(pool1, images, pool1)

        #grads_dict['conv2a'] = tf.gradients(conv2a, images, conv2a)
        #grads_dict['conv2b'] = tf.gradients(conv2b, images, conv2b)
        #grads_dict['pool2'] = tf.gradients(pool2, images, pool2)
        #
        #grads_dict['conv3a'] = tf.gradients(conv3a, images, conv3a)
        #grads_dict['conv3b'] = tf.gradients(conv3b, images, conv3b)
        #grads_dict['pool3'] = tf.gradients(pool3, images, pool3)
        #
        #grads_dict['conv4a'] = tf.gradients(conv4a, images, conv4a)
        #grads_dict['conv4b'] = tf.gradients(conv4b, images, conv4b)

        #grads_dict['convPa'] = tf.gradients(convPa, images, convPa)
        #grads_dict['convPb'] = tf.gradients(convPb, images, convPb)

        #grads_dict['convDa'] = tf.gradients(convDa, images, convDa)
        #grads_dict['convDb'] = tf.gradients(convDb, images, convDb)

    if GUIDED_BACKPROP:
        grads_dict['conv1a'] = tf.gradients(conv1a, images, conv1a)
        grads_dict['conv1a'] = grads_dict['conv1a']*(grads_dict['conv1a']>0)
        
        # d(conv1b)/d(conv1a) * [d(conv1b)/d(conv1a)>0] * [conv1a>0]
        grads_dict['conv1b'] = tf.gradients(conv1b, conv1a, conv1b)
        grads_dict['conv1b'] = grads_dict['conv1b']*(grads_dict['conv1b']>0)
        grads_dict['conv1b'] = grads_dict['conv1b'][0] * tf.cast((conv1a>0), tf.float32)
        # * d(conv1a)/d(images) * [d(conv1a)/d(images)>0] * [images>0]
        grads_dict['conv1b'] = tf.gradients(conv1a, images, grads_dict['conv1b'])
        grads_dict['conv1b'] = grads_dict['conv1b']*(grads_dict['conv1b']>0)


    #feat['conv1a'] = conv1a
    #feat['conv1b'] = conv1b
    #feat['pool1'] = pool1
    #feat['conv2a'] = conv2a
    #feat['conv2b'] = conv2b
    #feat['pool2'] = pool2
    #feat['conv3a'] = conv3a
    #feat['conv3b'] = conv3b
    #feat['pool3'] = pool3
    #feat['conv4a'] = conv4a
    #feat['conv4b'] = conv4b
    #feat['convPa'] = convPa
    #feat['convPb'] = convPb
    #feat['convDa'] = convDa
    feat['convDb'] = convDb

    return feat, grads_dict



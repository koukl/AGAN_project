import tensorflow as tf
import config
from ops import *

def generator_net(feed_var, g_bn0, g_bn1, g_bn2, g_bn3, reuse=False):
  with tf.variable_scope("generator") as scope:
    if reuse:
      scope.reuse_variables()
      Nbatchdata = config.sample_size
    else:
      Nbatchdata = config.batch_size
    if feed_var.shape[0] != Nbatchdata:
      raise ValueError('feed_var size: ', str(feed_var.shape[0]), ' not equal Nbatchdata: ', str(Nbatchdata))
    s_h, s_w = config.image_height, config.image_width
    s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
    s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
    s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
    s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

    h0 = tf.reshape(linear(feed_var, config.g_fm*8*s_h16*s_w16, name='g_h0_linear'), [-1, s_h16, s_w16, config.g_fm*8])
    if config.batchnorm:
      if reuse:
        h0 = g_bn0(h0, train=False)
      else:
        h0 = g_bn0(h0)
    h0 = tf.nn.relu(h0)
    h1 = deconv2d(h0, [Nbatchdata, s_h8, s_w8, config.g_fm*4], name='g_h1_conv')
    if config.batchnorm:
      if reuse:
        h1 = g_bn1(h1, train=False)
      else:
        h1 = g_bn1(h1)
    h1 = tf.nn.relu(h1)
    h2 = deconv2d(h1, [Nbatchdata, s_h4, s_w4, config.g_fm * 2], name='g_h2_conv')
    if config.batchnorm:
      if reuse:
        h2 = g_bn2(h2, train=False)
      else:
        h2 = g_bn2(h2)
    h2 = tf.nn.relu(h2)
    h3 = deconv2d(h2, [Nbatchdata, s_h2, s_w2, config.g_fm * 1], name='g_h3_conv')
    if config.batchnorm:
      if reuse:
        h3 = g_bn3(h3, train=False)
      else:
        h3 = g_bn3(h3)
    h3 = tf.nn.relu(h3)
    h4 = tf.nn.sigmoid(deconv2d(h3, [Nbatchdata, s_h, s_w, config.c_dim], name='g_h4_conv'))

    return h4



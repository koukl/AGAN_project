import tensorflow as tf
import numpy      as np
from   ops        import *
import math       as ma
import config
# define 
#   x = x:           real data
#   D(x) = dout:     discriminator score for real data, discriminator want this to be big
#   z = z:           generator noise 
#   G(z) = gout:     generated data
#   D(G(z)) = dgout: discriminator score for generated data, discriminator want this to be small
#                    generator want this to be big

# ----------------------------------------------------------------
def discriminator_net(feed_var,gan_type, d_bn1, d_bn2, d_bn3, reuse=False):
  with tf.variable_scope("discriminator") as scope:
    if reuse:
      scope.reuse_variables()

    h0 = lrelu(conv2d(feed_var, config.d_fm, name='d_h0_conv'))
    h1 = conv2d(h0, config.d_fm*2, name='d_h1_conv')
    if config.batchnorm:
      h1 = d_bn1(h1)
    h1 = lrelu(h1)
    h2 = conv2d(h1, config.d_fm*4, name='d_h2_conv')
    if config.batchnorm:
      h2 = d_bn2(h2)
    h2 = lrelu(h2)
    h3 = conv2d(h2, config.d_fm*8, name='d_h3_conv')
    if config.batchnorm:
      h3 = d_bn3(h3)
    h3 = lrelu(h3)
    #h4 = tf.nn.sigmoid(linear(tf.reshape(h3, [config.batch_size, -1]), 1, name='d_h4_linear'))
    Nnodes = np.prod(h3.get_shape().as_list()[1:])
    #h4 = linear(tf.reshape(h3, [config.batch_size, -1]), 1, name='d_h4_linear')
    h4 = linear(tf.reshape(h3, [-1, Nnodes]), 1, name='d_h4_linear')
    #h4 = linear(tf.reshape(h3, [bs, -1]), 1, name='d_h4_linear')

    return h4

'''
    # 10 fully connected layers
    layer_nodes = []
    layer_nodes.append(ndim)                    # 0 -- data layer
    layer_nodes.append(base_nodes)              # 1 -- 1st hidden layer
    layer_nodes.append(layer_nodes[1])          # 2
    layer_nodes.append(layer_nodes[2])          # 3
    layer_nodes.append(final_node)              # 13 -- output layer

    # declare the weights
    w = []
    b = []
    w.append("unused")  # to keep indexing consistent, make first index of weights unused
    b.append("unused")  # to keep indexing consistent, make first index of weights unused
    for i in range(1,len(layer_nodes)):
        w.append(tf.get_variable('d_w'+str(i), [layer_nodes[i-1],layer_nodes[i]], initializer=tf.random_normal_initializer(stddev=1)))
        #w.append(tf.Variable(tf.truncated_normal([layer_nodes[i-1],layer_nodes[i]], stddev=1), name='d_w'+str(i)))
        #b.append(tf.Variable(tf.constant(0.0, shape=[layer_nodes[i]]) ,name='d_b'+str(i)))
        b.append(tf.get_variable('d_b' + str(i), [layer_nodes[i]], initializer=tf.zeros_initializer()))

    # add the layers
    a = []
    a.append("unused") # to keep indexing consistent a[0] is unused
    a.append(tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(feed_var , w[1]), b[1]), name="h1"))
    for i in range(2,len(w)-1):
      a.append(tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(a[i-1], w[i]), b[i]), name="h"+str(i)))

    last_a = a[len(a)-1]
    last_w = w[len(w)-1]
    last_b = b[len(b)-1]
    dout = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(last_a, last_w), last_b))

    return dout
'''

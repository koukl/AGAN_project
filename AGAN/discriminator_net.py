import tensorflow as tf
import numpy      as np
import math       as ma

# define 
#   x = x:           real data
#   D(x) = dout:     discriminator score for real data, discriminator want this to be big
#   z = z:           generator noise 
#   G(z) = gout:     generated data
#   D(G(z)) = dgout: discriminator score for generated data, discriminator want this to be small
#                    generator want this to be big

# ----------------------------------------------------------------
def discriminator_net(feed_var,gan_type):

  ndim       = 2
  final_node = 1
  base_nodes = 10
  
  # 10 fully connected layers 
  layer_nodes = []
  layer_nodes.append(ndim)                    # 0 -- data layer
  layer_nodes.append(base_nodes)              # 1 -- 1st hidden layer
  layer_nodes.append(layer_nodes[1])          # 2
  layer_nodes.append(layer_nodes[2])          # 3
#  layer_nodes.append(layer_nodes[3])          # 4
#  layer_nodes.append(layer_nodes[4])          # 5
#  layer_nodes.append(layer_nodes[5])          # 6
#  layer_nodes.append(layer_nodes[6])          # 7
#  layer_nodes.append(layer_nodes[7])          # 8
#  layer_nodes.append(layer_nodes[8])          # 9
#  layer_nodes.append(layer_nodes[9])          # 10 
#  layer_nodes.append(layer_nodes[10])         # 11
#  layer_nodes.append(layer_nodes[11])         # 12
  layer_nodes.append(final_node)              # 13 -- output layer
  
  # declare the weights
  w = []
  b = []
  w.append("unused")  # to keep indexing consistent, make first index of weights unused
  b.append("unused")  # to keep indexing consistent, make first index of weights unused
  for i in range(1,len(layer_nodes)):
      w.append(tf.Variable(tf.truncated_normal([layer_nodes[i-1],layer_nodes[i]], stddev=1)))
      b.append(tf.Variable(tf.constant(0.0, shape=[layer_nodes[i]])))
  
  dout = rebuild_net(feed_var,w,b,gan_type)

  return(w,b,dout)

# ----------------------------------------------------------------
def rebuild_net(feed_var,w,b,gan_type):

  # add the layers
  a = []
  a.append("unused") # to keep indexing consistent a[0] is unused
  a.append(tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(feed_var , w[1]), b[1]), name="h1"))
  for i in range(2,len(w)-1):
    a.append(tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(a[i-1], w[i]), b[i]), name="h"+str(i)))
  
  last_a = a[len(a)-1]
  last_w = w[len(w)-1]
  last_b = b[len(b)-1]

  if gan_type == "gan":
    dout = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(last_a, last_w), last_b))
  elif gan_type == "wgan":
    # don't restrict output range if using wgan
    dout = tf.nn.bias_add(tf.matmul(last_a, last_w), last_b)
 
  return(dout)

import tensorflow as tf
import numpy      as np
import math       as ma
import pandas     as pd
import sys
from visualize import *

# define 
#   x = x:           real data
#   D(x) = dout:     discriminator score for real data, discriminator want this to be big
#   z = z:           generator noise 
#   G(z) = gout:     generated data
#   D(G(z)) = dgout: discriminator score for generated data, discriminator want this to be small
#                    generator want this to be big

def optimize_gan(sess,gan_type,trainD,trainWD,trainG,zmean,zstdd,xpos,z,xfeed,clipper,i):
  GANN   = 1
  DISN   = 1
  ZN     = 500
  eachturn = 10


# -----------------------------------------------------------
# do the discriminator, generator adversarial
# -----------------------------------------------------------

'''
#   generate the negative data
  zfeed = tf.truncated_normal([ZN,2],mean=zmean,stddev=zstdd).eval()

  dgfeed = {xpos:xfeed, z:zfeed}

#   train the generator
  if (i/eachturn)%2 == 1:
    print('g')
    for g in range(GANN):
      sess.run(trainG,feed_dict=dgfeed)

#   train the discriminator
  if (i/eachturn)%2 == 0:
    print('d')
    for d in range(DISN):
      if gan_type=="wgan":
        sess.run(trainWD,feed_dict=dgfeed)
        sess.run(clipper)
      else:
        for repeat in range(8):
          sess.run(trainD,feed_dict=dgfeed)

  return(zfeed)
'''

#   generate the negative data
#
  zfeed = tf.truncated_normal([ZN,2],mean=zmean,stddev=zstdd).eval()

  dgfeed = {xpos:xfeed, z:zfeed}

#   train the generator
  if i != 0:
    for g in range(GANN):
      sess.run(trainG,feed_dict=dgfeed)

#   train the discriminator
  for d in range(DISN):
    if gan_type=="wgan":
      sess.run(trainWD,feed_dict=dgfeed)
      sess.run(clipper)
    else:
      sess.run(trainD,feed_dict=dgfeed)

  return(zfeed)
 

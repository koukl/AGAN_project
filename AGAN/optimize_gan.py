import tensorflow as tf
import numpy      as np
import math       as ma
import pandas     as pd
import sys
from visualize import *
import config

# define
#   x = x:           real data
#   D(x) = dout:     discriminator score for real data, discriminator want this to be big
#   z = z:           generator noise 
#   G(z) = gout:     generated data
#   D(G(z)) = dgout: discriminator score for generated data, discriminator want this to be small
#                    generator want this to be big

def optimize_gan(sess, gan_type, trainD, trainG, trainWD, trainWG, xpos, z, xfeed, clipper,gout,accum_gout,accum_gout_,i):

    # -----------------------------------------------------------
    # do the discriminator, generator adversarial
    # -----------------------------------------------------------

    #   generate the negative data
    #zfeed = tf.truncated_normal([ZN, 2], mean=zmean, stddev=zstdd).eval()
    zfeed = np.random.uniform(0, 1, size=(config.ZN, 2))
    dgfeed = {xpos: xfeed, z: zfeed}

    #   train the generator
    if i > 0:
        for g in range(config.N_G):
            if gan_type == "wgan":
                sess.run(trainWG, feed_dict=dgfeed)
            else:
                sess.run(trainG, feed_dict=dgfeed)

    gout_ = sess.run(gout, feed_dict=dgfeed)

    # retain beta fraction of the old samples
    np.random.shuffle(accum_gout_)
    Nretain = int(config.beta * accum_gout_.shape[0])
    all_accum_gout_ = np.append(accum_gout_[:Nretain], gout_, axis=0)

    #   train the discriminator
    for d in range(config.N_D):
        if gan_type == "wgan":
            dgoutfeed = {xpos: xfeed, z: zfeed, accum_gout: accum_gout_}
            sess.run(trainWD, feed_dict=dgoutfeed)
            sess.run(clipper)
        else:
            dgoutfeed = {xpos: xfeed, z: zfeed, accum_gout: accum_gout_}
            sess.run(trainD, feed_dict=dgoutfeed)

    return all_accum_gout_


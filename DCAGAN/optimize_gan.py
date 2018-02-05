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

def optimize_gan(sess, gan_type, trainD, trainG, trainWD, trainWG, xpos, z, batch_images, gout,accum_gout,accum_gout_):

    # -----------------------------------------------------------
    # do the discriminator, generator adversarial
    # -----------------------------------------------------------

    #   generate the negative data
    zfeed = np.random.uniform(-1, 1, [config.batch_size, config.z_dim]).astype(np.float32)
    dgfeed = {xpos: batch_images, z: zfeed}

    #   train the generator
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
            dgoutfeed = {xpos: batch_images, z: zfeed, accum_gout: all_accum_gout_}
            sess.run(trainWD, feed_dict=dgoutfeed)
            sess.run(clipper)
        else:
            dgoutfeed = {xpos: batch_images, z: zfeed, accum_gout: all_accum_gout_}
            sess.run(trainD, feed_dict=dgoutfeed)

    return all_accum_gout_

import tensorflow as tf
import numpy      as np
import math       as ma
import pandas     as pd
import sys
from visualize         import *
#from get_xfeed         import *
from optimize_gan      import *
from discriminator_net import *
from generator_net     import *
from loss_functions    import *
from load_data         import *
from ops               import *
import config
from visualize          import *

# define 
#   x = x:           real data
#   D(x) = dout:     discriminator score for real data, discriminator want this to be big
#   z = z:           generator noise 
#   G(z) = gout:     generated data
#   D(G(z)) = dgout: discriminator score for generated data, discriminator want this to be small
#                    generator want this to be big

np.random.seed(config.seed)

data_X = load_data()
config.c_dim = data_X[0].shape[-1]
config.grayscale = (config.c_dim == 1)

with tf.Session() as sess:

  tf.set_random_seed(config.seed)
  xpos = tf.placeholder(tf.float32, [config.batch_size] + config.image_dims, name="xpos")
  z = tf.placeholder(tf.float32, [None, config.z_dim], name="z")

  # batch normalization : deals with poor initialization helps gradient flow
  d_bn1 = batch_norm(name='d_bn1')
  d_bn2 = batch_norm(name='d_bn2')
  d_bn3 = batch_norm(name='d_bn3')

  g_bn0 = batch_norm(name='g_bn0')
  g_bn1 = batch_norm(name='g_bn1')
  g_bn2 = batch_norm(name='g_bn2')
  g_bn3 = batch_norm(name='g_bn3')

  # -----------------------------------------------------------
# discriminator network - (im_h * im_w) -> conv . . .  -> 1
# -----------------------------------------------------------
  dout = discriminator_net(xpos, config.gan_type, d_bn1, d_bn2, d_bn3)
# -----------------------------------------------------------
# generator network - 2 -> transpose_conv . . . -> (im_h * im_w)
# -----------------------------------------------------------
  gout = generator_net(z, g_bn0, g_bn1, g_bn2, g_bn3)
  accum_gout = tf.placeholder(tf.float32, [None] + config.image_dims, name="accum_gout")
  dgout_current = discriminator_net(gout,config.gan_type,d_bn1, d_bn2, d_bn3, reuse=True)
  dgout_past = discriminator_net(accum_gout, config.gan_type, d_bn1, d_bn2, d_bn3, reuse=True)
  t_vars = tf.trainable_variables()
  d_vars = [var for var in t_vars if 'd_' in var.name]
  g_vars = [var for var in t_vars if 'g_' in var.name]
# -----------------------------------------------------------
# loss functions
# -----------------------------------------------------------
  if config.gan_type == "gan":
    dloss = gandloss(dout, dgout_current, dgout_past)
    gloss = gangloss(dgout_current)
  else:
    dloss = wgandloss(dout, dgout_current)
    gloss = wgangloss(dgout_current)

  trainD  = tf.train.AdamOptimizer(learning_rate=config.d_lr, beta1=config.adam_param).minimize(dloss,name="trainD",var_list=d_vars)
  trainWD = tf.train.RMSPropOptimizer(0.01).minimize(dloss,name="trainWD",var_list=d_vars)
  trainG  = tf.train.AdamOptimizer(learning_rate=config.g_lr, beta1=config.adam_param).minimize(gloss,name="trainG",var_list=g_vars)
  trainWG = tf.train.RMSPropOptimizer(0.01).minimize(gloss, name="trainWG", var_list=g_vars)

  init = tf.variables_initializer(tf.global_variables(), name='init_all_vars_op')
  sess.run(init)

# -----------------------------------------------------------
# do the discriminator, generator adversarial
# -----------------------------------------------------------
  alldloss = np.empty((0,1))
  allgloss = np.empty((0,1))
  accum_gout_ = np.zeros(tuple([0])+ tuple(config.image_dims))
  Nbatches = min(len(data_X), config.train_size) // config.batch_size
  sample_zfeed = np.random.uniform(-1, 1, [config.sample_size, config.z_dim]).astype(np.float32)
  count = 0
  for i in range(config.Nepoch):
    for idx in range(Nbatches):
      batch_images = data_X[idx * config.batch_size:(idx + 1) * config.batch_size]
      batch_z = np.random.uniform(-1, 1, [config.batch_size, config.z_dim]).astype(np.float32)
      accum_gout_ = optimize_gan(sess, config.gan_type, trainD, trainG, trainWD, trainWG, xpos, z, batch_images, gout,
                                 accum_gout, accum_gout_)
      if count % config.print_iter == 0:
        chosen_samples = np.arange(config.train_size)
        np.random.shuffle(chosen_samples)
        chosen_samples = chosen_samples[:config.sample_size]
        sample_images = data_X[chosen_samples]
        dgoutfeed = {xpos: sample_images, z: sample_zfeed, accum_gout: accum_gout_}
        samples, curdloss, curgloss = sess.run([gout, dloss, gloss], feed_dict=dgoutfeed)
        print("# %s after D: dloss:%s gloss:%s" % (count, curdloss, curgloss))
        sys.stdout.flush()
        alldloss = np.append(alldloss, np.array([[curdloss]]), axis=0)
        allgloss = np.append(allgloss, np.array([[curgloss]]), axis=0)
        # samples = sample_images
        # crop samples to Ncropsamples
        samples = samples[:config.Ncropsamples]
        save_images(samples, image_manifold_size(samples.shape[0]),
                    '{}/train_{:04d}.png'.format(config.sample_dir, count))
        save_loss(alldloss, allgloss,'{}/loss_{:04d}.png'.format(config.sample_dir, count) )

      count += 1



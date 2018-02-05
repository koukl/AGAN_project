import tensorflow as tf
import numpy      as np
import math       as ma
import pandas     as pd
import sys
from visualize         import *
from get_xfeed         import *
from optimize_gan      import *
from discriminator_net import *
from generator_net     import *
from loss_functions    import *
import config

# define 
#   x = x:           real data
#   D(x) = dout:     discriminator score for real data, discriminator want this to be big
#   z = z:           generator noise 
#   G(z) = gout:     generated data
#   D(G(z)) = dgout: discriminator score for generated data, discriminator want this to be small
#                    generator want this to be big

np.random.seed(config.seed)

with tf.Session() as sess:

  tf.set_random_seed(config.seed)

  real_data = np.loadtxt('./data/' + config.data + '_data.txt', delimiter=',')
  xpos = tf.placeholder(tf.float32, [None, 2], name="xpos")
  z = tf.placeholder(tf.float32, [None, 2], name="z")

  print("gan algorithm use  : ",config.gan_type)

# -----------------------------------------------------------
# generate the positive data 
# -----------------------------------------------------------
  #xfeed = get_xfeed(n1,xmean1,xstdd1,n2,xmean2,xstdd2)
  xfeed = real_data
# -----------------------------------------------------------
# discriminator network - 2 -> nhidden . . .  -> 1
# -----------------------------------------------------------
  wd,bd,dout = discriminator_net(xpos,config.gan_type)

  # ck: discrminator network on grid
  gridxy   = tf.placeholder(tf.float32, [None, 2], name="gridxy")
  dgridout = rebuild_net(gridxy,wd,bd,config.gan_type)

  var_list_d = wd[1:]+bd[1:]
  clipper = [v.assign(tf.clip_by_value(v,-config.clipw,config.clipw)) for v in var_list_d]
# -----------------------------------------------------------
# generator network - 2 -> nhidden . . . -> 2
# -----------------------------------------------------------
  wg,bg,gout, h1, h2 = generator_net(z)
  accum_gout = tf.placeholder(tf.float32, [None, 2], name="accum_gout")
  dgout_current = rebuild_net(gout,wd,bd,config.gan_type)
  dgout_past = rebuild_net(accum_gout, wd, bd, config.gan_type)
  var_list_g = wg[1:]+bg[1:]
# -----------------------------------------------------------
# loss functions
# -----------------------------------------------------------
  if config.gan_type == "gan":
    dloss  = gandloss(dout, dgout_current, dgout_past) + config.d_l2_reg * tf.reduce_sum([tf.nn.l2_loss(w) for w in wd[1:]])
    gloss  = gangloss(dgout_current)
  else:
    dloss = wgandloss(dout, dgout_current, dgout_past) + config.d_l2_reg * tf.reduce_sum([tf.nn.l2_loss(w) for w in wd[1:]])
    gloss = wgangloss(dgout_current)

  trainD  = tf.train.AdamOptimizer(learning_rate=config.d_lr).minimize(dloss,name="trainD",var_list=var_list_d)
  trainWD = tf.train.RMSPropOptimizer(learning_rate=config.d_lr).minimize(dloss,name="trainWD",var_list=var_list_d)
  trainG  = tf.train.AdamOptimizer(learning_rate=config.g_lr).minimize(gloss,name="trainG",var_list=var_list_g)
  trainWG = tf.train.RMSPropOptimizer(learning_rate=config.g_lr).minimize(gloss, name="trainWG", var_list=var_list_g)

  init = tf.variables_initializer(tf.global_variables(), name='init_all_vars_op')
  sess.run(init)

# -----------------------------------------------------------
# do the discriminator, generator adversarial
# -----------------------------------------------------------
  alldloss = np.empty((0,1))
  allgloss = np.empty((0,1))
  accum_gout_ = np.zeros((0,2))
  for i in range(config.Niter):
    accum_gout_ = optimize_gan(sess,config.gan_type,trainD,trainG,trainWD,trainWG,xpos,z,xfeed,clipper,gout,accum_gout, accum_gout_,i)

    if i % config.print_iter==0:
      zfeed = np.random.uniform(0, 1, size=(500, 2))
      curdloss, curgloss = sess.run([dloss, gloss],feed_dict={xpos:xfeed, z:zfeed, accum_gout:accum_gout_})
      print("# %s after D: dloss:%s gloss:%s"%(i,curdloss,curgloss))
      max_dw = []
      min_dw = []
      for vard in var_list_d:
        max_dw.append(np.max(vard.eval()))
        min_dw.append(np.min(vard.eval()))
      if config.gan_type == "wgan":
        print("# Clipping: %s, max D parameter: %s, min D parameter:%s"%(config.clipw,np.max(max_dw), np.min(min_dw)))
      sys.stdout.flush()
      alldloss = np.append(alldloss, np.array([[curdloss]]), axis=0)
      allgloss = np.append(allgloss, np.array([[curgloss]]), axis=0)
      visualize(gout, h1, h2, dgridout, xpos, gridxy, z, zfeed, xfeed, sess, i, alldloss, allgloss,config.pltshow, accum_gout_)
      #visualize_manifold_hidden3_x1_indiv(gout,h1,h2,dgridout,xpos,gridxy,z,zfeed,xfeed,sess, i, alldloss, allgloss,pltshow)


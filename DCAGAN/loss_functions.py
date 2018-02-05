import tensorflow as tf

# -------------------------------------------------------------
def gandloss(dout, dgout_current, dgout_past):

  #meand1 = ( -1.0*tf.reduce_mean(tf.log(       dout + 1e-8 )) )
  #meand2 = ( -1.0*tf.reduce_mean(tf.log(1.0 - dgout + 1e-8 )) )

  meand1 =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dout, labels=tf.ones_like(dout)))
  meand2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dgout_current, labels=tf.zeros_like(dgout_current)))
  meand3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dgout_past, labels=tf.zeros_like(dgout_past)))

  return( meand1 + 0.5 * (meand2 + meand3) )
# -------------------------------------------------------------
def gangloss(dgout):
  #meand2 = ( tf.reduce_mean(tf.log(1.0 - dgout + 1e-8 )) )
  meand2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dgout, labels=tf.ones_like(dgout)))

  return(meand2)

# -------------------------------------------------------------
def wgandloss(dout,dgout):

  meand1 = (-1.0*tf.reduce_mean(dout ) )
  meand2 = (     tf.reduce_mean(dgout) )

  return( meand1 + meand2 )

# -------------------------------------------------------------
def wgangloss(dgout):

  meand2 = ( -1.0*tf.reduce_mean(dgout) )

  return( meand2 )

# -------------------------------------------------------------

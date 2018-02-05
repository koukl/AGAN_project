import tensorflow as tf

# -------------------------------------------------------------
def gandloss(dout,dgout_current, dgout_past):

  meand1 = ( -1.0*tf.reduce_mean(tf.log(       dout + 1e-8 )) )
  meand2 = ( -1.0*tf.reduce_mean(tf.log(1.0 - dgout_current + 1e-8 )) )
  meand3 = (-1.0 * tf.reduce_mean(tf.log(1.0 - dgout_past + 1e-8)))

  return( meand1 + 0.5 * (meand2 + meand3) )

# -------------------------------------------------------------
def gangloss(dgout):
  meand2 = ( tf.reduce_mean(tf.log(1.0 - dgout + 1e-8 )) )
  return(meand2)

# -------------------------------------------------------------
def wgandloss(dout,dgout_current, dgout_past):

  meand1 = (-1.0*tf.reduce_mean(dout ) )
  meand2 = (     tf.reduce_mean(dgout_current) )
  meand3 = (     tf.reduce_mean(dgout_past)    )

  return( meand1 + 0.5 * (meand2 + meand3) )

# -------------------------------------------------------------
def wgangloss(dgout):

  meand2 = ( -1.0*tf.reduce_mean(dgout) )

  return( meand2 )

# -------------------------------------------------------------

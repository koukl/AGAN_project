import tensorflow as tf
import numpy      as np

def get_xfeed(n1,xmean1,xstdd1,n2,xmean2,xstdd2):

  x1 = tf.truncated_normal([n1,2],mean=xmean1,stddev=xstdd1).eval()
  x2 = tf.truncated_normal([n2,2],mean=xmean2,stddev=xstdd2).eval()
  
  return(np.concatenate((x1,x2)))


import tensorflow as tf
import numpy as np

def var_rn(shape, mean=0.0, stddev=0.1):
  return tf.Variable(tf.random_normal(shape, mean, stddev))

def one_hot(array, n_output):
  return np.reshape(np.eye(n_output)[array], (-1, n_output))

def get_xy(data, n_output):
  return data[:,1:], one_hot(np.reshape(data[:,0], (-1,1)).astype(np.int32), n_output)

def get_xy_c(data, n_output):
  return data[:,:-1], one_hot(np.reshape(data[:,-1], (-1,1)).astype(np.int32), n_output)

def next_batch(data, step, bpe, batch_size=100):
  step %= bpe
  return data[step * batch_size : (step + 1) * batch_size, :]
def log(array):
  return tf.log(tf.clip_by_value(array, 1e-10, 1.5))

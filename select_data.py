import numpy as np
import cube_nn_p2 as nn
import tensorflow as tf

sess = tf.Session(graph=nn.graph)
nn.saver.restore(sess, "cubennp2.ckpt")

full_data = np.genfromtxt("data/p2_h_data.txt", max_rows = 300000)
def pred(inputs):
  inputs = np.reshape(inputs, [1,-1])
  return sess.run(nn.pred, feed_dict={nn.x:inputs, nn.y:np.array([[1]]), nn.keep_prob: 1.0})
g = []
p = []
for row in full_data:
  problematic_data = (np.absolute(row[-1] - pred(row[:-1]))) > 7# full data where abs(data[-1] - pred ) > k)
  g.append(problematic_data)
  if (bool(problematic_data)):
    p.append(row[-1]) 
mask = np.reshape(np.array(g), [-1,])
data = (full_data[mask, :]).astype(np.int32)
print data.shape
f = open("data/difficult_data.txt", 'w')
print np.mean(np.array(p))
for row in data:
  for element in row:
    f.write(str(element))
    f.write(" ")
  f.write("\n")

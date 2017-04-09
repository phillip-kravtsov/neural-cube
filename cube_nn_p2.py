import time
import numpy as np
import tensorflow as tf
import nn_utils as utils

graph = tf.Graph()

with graph.as_default():
  k = 2.0
  learning_rate = .001
  steps = 20000 + 1
  batch_size = 64
  n_input = 288
  n_output = 1

  n_fc1 = 300
  n_fc2 = 200
  n_fc3 = 100
  dropout = 1.0

  x = tf.placeholder(tf.float32, [None, n_input])
  y = tf.placeholder(tf.float32, [None, n_output])
  keep_prob = tf.placeholder(tf.float32)

  def net(x, w, b):
    fc1 = tf.matmul(x, w['w1']) +  b['b1']
    fc1 = tf.nn.relu(fc1)

    fc2 = tf.nn.relu(tf.matmul(fc1, weights['w2']) + b['b2'])

    fc3 = tf.nn.relu(tf.matmul(fc2, weights['w3']) + b['b3'])

    out = tf.matmul(fc3, weights['out']) + biases['bout']

    return out

  weights = {
    'w1' : utils.var_rn([n_input, n_fc1]),
    'w2' : utils.var_rn([n_fc1, n_fc2]),
    'w3' : utils.var_rn([n_fc2, n_fc3]),
    'out': utils.var_rn([n_fc3, n_output]),

  }

  biases = {
    'b1' : utils.var_rn([n_fc1]),
    'b2' : utils.var_rn([n_fc2]),
    'b3' : utils.var_rn([n_fc3]),
    'bout': utils.var_rn([n_output]),
  }

  pred = net(x, weights, biases)
  cost = tf.reduce_mean(tf.square(pred-y) + k * tf.nn.relu(pred-y))

  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

  delta = pred - y
  correct_pred = tf.less_equal(tf.abs(delta), 2.0)
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

  display_step = 1000
  t_display = 1000
  init = tf.global_variables_initializer()
  saver = tf.train.Saver()

  def next_batch(data, step):
    bpe = data.shape[0] / batch_size
    step %= bpe
    return data[step * batch_size: (step + 1) * batch_size, :]

  def load_data():
    global full_data, train_data, test_data, val_data, train_data
    full_data = np.genfromtxt("data/p2_h_data.txt", max_rows = 220000)
    print("Data retrieved")

    np.random.seed(2718)
    np.random.shuffle(full_data)

    print(full_data.shape)
    train_data, test_data = np.split(full_data, [(95 * full_data.shape[0])/100])
    val_data, train_data = np.split(train_data, [(2 * train_data.shape[0])/100])

  def train(write, get):
    with tf.Session(graph=graph) as sess:
      load_data()
      if(True):
        if(get):
          saver.restore(sess, 'cubennp2.ckpt')
        else:
          sess.run(init)
        step = 1
        np.random.shuffle(train_data)

        while (step < steps):
          bpe = train_data.shape[0] / batch_size
          if (step % bpe == 0):
            np.random.shuffle(train_data)

          n_b = next_batch(train_data, step)
          batch_x, batch_y = n_b[:,:-1], np.reshape(n_b[:,-1], (n_b.shape[0], 1))
          sess.run(optimizer, feed_dict={x:batch_x, y:batch_y, keep_prob: dropout})

          if (step % display_step == 0):
            val_x, val_y = val_data[:,:-1], np.reshape(val_data[:,-1],( val_data.shape[0],1))
            loss, acc, d = sess.run([cost, accuracy, delta], feed_dict={x:val_x, y:val_y, keep_prob:1.0})

            print("Step: %d, Loss: %f, Accuracy: %f"%(step, loss, acc))

            if (loss < 1.2 and write):
              saver.save(sess, "cubennp2.ckpt")
              return
          if (step % t_display == 0):
            t_x, t_y = train_data[:10000,:-1], np.reshape(train_data[:10000,-1], (10000,1))
            t_loss, t_acc = sess.run([cost, accuracy], feed_dict = {x:t_x, y:t_y, keep_prob:1.0})
            print("tloss: %f, tacc: %f "%(t_loss, t_acc))
            print("\n")

        #val_x, val_y = val_data[:,:-1], np.reshape(val_data[:,-1], (val_data.shape[0],1))
          step+= 1

        if (write):
          print("Writing in 10 seconds.")
          time.sleep(10)
          print("Writing.")
          saver.save(sess, 'cubennp2.ckpt')

 # train(True, True)
#sess = tf.Session()
#s.restore(sess, 'cubennp2.ckpt')
#print "everything is totally fine"
#save_path = s.save(sess, 'cubennp2.ckpt')

#print sess.run(weights['w2'])
#load_data()
#print sess.run(worst,feed_dict={x:val_data[:,:-1], y:np.reshape(val_data[:,-1], [-1,1]), keep_prob:1.0})

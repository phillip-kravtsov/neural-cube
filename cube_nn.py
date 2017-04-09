import time
import numpy as np
import tensorflow as tf


graph = tf.Graph()

with graph.as_default():
  k = 2.0
  learning_rate = .001
  steps = 20000 + 1
  batch_size = 64
  n_input = 60
  n_output = 1

  n_fc1 = 256
  n_fc2 = 128
  n_fc3 = 64


  dropout = 1.0

  x = tf.placeholder(tf.float32, [None, n_input])
  y = tf.placeholder(tf.float32, [None, n_output])
  keep_prob = tf.placeholder(tf.float32)

  def net(x, w, b):
    fc1 = tf.matmul(x, w['w1']) +  b['b1']
    fc1 = tf.nn.relu(fc1)

    fc2 = tf.nn.relu(tf.matmul(fc1, weights['w2']) + b['b2'])
    fc2 = tf.nn.dropout(fc2, keep_prob)

    fc3 = tf.nn.relu(tf.matmul(fc2, weights['w3']) + b['b3'])

    out = tf.matmul(fc3, weights['out']) + biases['bout']

    return out

  with tf.variable_scope("p1net", initializer=tf.random_normal_initializer(0.1,0.1)):

    weights = {
      'w1' : tf.get_variable('w1', [n_input, n_fc1]),
      'w2' : tf.get_variable('w2', [n_fc1, n_fc2]),
      'w3' : tf.get_variable('w3', [n_fc2, n_fc3]),
      'out': tf.get_variable('out', [n_fc3, n_output]),
    }

    biases = {
      'b1' : tf.get_variable('b1', [n_fc1]),
      'b2' : tf.get_variable('b2', [n_fc2]),
      'b3' : tf.get_variable('b3', [n_fc3]),
      'bout': tf.get_variable('bout',[n_output]),
    }

  pred = net(x, weights, biases)
  cost = tf.reduce_mean(tf.square(pred-y) + k * tf.nn.relu(pred-y))

  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

  delta = pred - y
  correct_pred = tf.equal(tf.round(pred), y)
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
    np.random.seed(12)
    global full_data, train_data, test_data, val_data, train_data
    full_data = np.genfromtxt("data/p1_with_edges.txt")
    np.random.shuffle(full_data)

    train_data, test_data = np.split(full_data, [(99 * full_data.shape[0])/100])
    val_data, train_data = np.split(train_data, [(10 * train_data.shape[0])/100])

  def train(write, get):
    load_data()
    with tf.Session(graph=graph) as sess:
      if(get):
        saver.restore(sess, 'edgycubenn3.ckpt')

      else:
        sess.run(init)

      step = 1
      while (step < steps):

        global learning_rate
        bpe = train_data.shape[0] / batch_size

        if (step % bpe == 0):
          np.random.shuffle(train_data)

        n_b = next_batch(train_data, step)
        batch_x, batch_y = n_b[:,:-1], np.reshape(n_b[:,-1], (n_b.shape[0], 1))
        sess.run(optimizer, feed_dict={x:batch_x, y:batch_y, keep_prob: dropout})

        if (step % display_step == 0):
          val_x, val_y = val_data[:,:-1], np.reshape(val_data[:,-1],( val_data.shape[0],1))
          loss, acc = sess.run([cost, accuracy], feed_dict={x:val_x, y:val_y, keep_prob:1.0})

          if (loss < 1.03 and write):
            print("Writing in 3 seconds.")
            time.sleep(3)
            print("Writing.")
            save_path = saver.save(sess, 'edgycubenn3.ckpt')
            return

          if (step % t_display == 0):
            t_x, t_y = train_data[:10000,:-1], np.reshape(train_data[:10000,-1], (10000,1))
            t_loss, t_acc = sess.run([cost, accuracy], feed_dict = {x:t_x, y:t_y, keep_prob:1.0})
            print("tloss: %f, tacc: %f "%(t_loss, t_acc))

          print("Step: %d , Loss: %f , Accuracy: %f "%(step, loss, acc))
          print(sess.run(pred, feed_dict={x:val_x[:10], y:val_y[:10], keep_prob: 1.0}))

          for i in xrange(10):
            print(delta.eval(feed_dict={x:np.reshape(val_x[i],[1,n_input]), y:np.reshape(val_y[i],[1,1]), keep_prob: 1.0}))

        step += 1
      if (write):
        print("Writing in 5 seconds.")
        time.sleep(5)
        print("Writing.")
        save_path = saver.save(sess, 'edgycubenn3.ckpt')

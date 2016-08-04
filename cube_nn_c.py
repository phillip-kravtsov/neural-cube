
import numpy as np
import tensorflow as tf

graph = tf.Graph()
with graph.as_default():

  learning_rate = 0.002
  steps = 10000
  batch_size = 128
  n_input = 298
  n_output = 10

  n_fc1 = 400
  n_fc2 = 300
  n_fc3 = 200
  n_fc4 = 200
  n_fc5 = 100
  dropout = 0.50

  initial_prev_move = (np.eye(10) * -10.0).astype(np.float32)
  x = tf.placeholder(tf.float32, [None, n_input])
  y = tf.placeholder(tf.float32, [None, n_output])
  keep_prob = tf.placeholder(tf.float32)

  def net(x, w, b):
    fc1 = tf.nn.relu(tf.matmul(x[:,:n_input-n_output], w['w1']) + b['b1'])
    #fc1 = tf.nn.dropout(fc1, keep_prob + 0.1)
    fc2 = tf.nn.relu(tf.matmul(fc1, w['w2']) + b['b2'])
    #fc2 = tf.nn.dropout(fc2, keep_prob)
    fc3 = tf.nn.relu(tf.matmul(fc2, w['w3']) + b['b3'])
    fc3 = tf.nn.dropout(fc3, keep_prob)
    fc4 = tf.nn.relu(tf.matmul(fc3, w['w4']) + b['b4'])
    fc5 = tf.nn.relu(tf.matmul(fc4, w['w5']) + b['b5'])
    fc5 = tf.nn.dropout(fc5, keep_prob)
    out = tf.matmul(fc5, weights['out']) + biases['out'] + tf.matmul(x[:,n_input - n_output:], weights['prev_move'])
    return tf.nn.softmax(out)

  weights = {
    'w1' : tf.Variable(tf.truncated_normal([n_input-10, n_fc1], stddev=0.1)),
    'w2' : tf.Variable(tf.truncated_normal([n_fc1, n_fc2], stddev=0.1)),
    'w3' : tf.Variable(tf.truncated_normal([n_fc2, n_fc3], stddev=0.1)),
    'w4' : tf.Variable(tf.truncated_normal([n_fc3, n_fc4], stddev=0.1)),
    'w5' : tf.Variable(tf.truncated_normal([n_fc4, n_fc5], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([n_fc5, n_output], stddev=0.1)),
    'prev_move': tf.Variable(initial_prev_move),
  }
  biases = {
    'b1' : tf.Variable(tf.truncated_normal([n_fc1], stddev=0.1)),
    'b2' : tf.Variable(tf.truncated_normal([n_fc2], stddev=0.1)),
    'b3' : tf.Variable(tf.truncated_normal([n_fc3], stddev=0.1)),
    'b4' : tf.Variable(tf.truncated_normal([n_fc4], stddev=0.1)),
    'b5' : tf.Variable(tf.truncated_normal([n_fc5], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([n_output], stddev=0.1)),
  }

  pred = net(x, weights, biases)
  correct_pred = tf.equal(tf.argmax(y,1), tf.argmax(pred,1))
  cross_entropy = tf.reduce_mean(-(y*tf.log(tf.clip_by_value(pred, 1e-10, 1.0))))

  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
  correct_top = tf.nn.in_top_k(pred, tf.cast(tf.argmax(y,1), tf.int32), 4)
  top_k_acc = tf.reduce_mean(tf.cast(correct_top, tf.float32))
  display_step = 1000

  init = tf.initialize_all_variables()
  saver = tf.train.Saver()
  def one_hot(array):
    return np.reshape(np.eye(n_output)[array], [-1,n_output])

  def next_batch(data, step):
    bpe = data.shape[0] / batch_size
    step %= bpe
    return data[step * batch_size : (step+1) * batch_size, :]

  def load_data():
    global full_data, train_data, test_data, val_data, train_data
    full_data = np.genfromtxt("data/fdata2.txt", max_rows=300000)
    np.random.shuffle(full_data)
    print full_data.shape
    train_data, test_data = np.split(full_data, [(95 * full_data.shape[0])/100])
    print train_data
    val_data, train_data = np.split(train_data, [(10 * train_data.shape[0])/100])

  def get_xy(data):
    return data[:,:0], one_hot(np.reshape(data[:,-1], (-1,1)).astype(np.int32))
  def train(get, write):
    load_data()
    with tf.Session(graph=graph) as sess:
      if (get):
        saver.restore(sess, "mvnet1.ckpt")
      else: 
        sess.run(init)  
      step = 1
      np.random.shuffle(train_data)
      bpe = train_data.shape[0] / batch_size
      while (step < steps):
      
        global learning_rate
        if (step == 5000):
          learning_rate = .001
          pass
        if (step % bpe == 0):
          print "step: %d , shuffling"%(step)
          np.random.shuffle(train_data)
        
      
        n_b = next_batch(train_data, step)
        b_x, b_y = get_xy(n_b)
        #print b_x 
        sess.run(optimizer, feed_dict={x:b_x, y:b_y, keep_prob:dropout})

        if (step % display_step == 1):
          val_x, val_y = get_xy(val_data)
          loss, acc, topk = sess.run([cross_entropy, accuracy, top_k_acc], feed_dict = {x:val_x, y:val_y, keep_prob: 1.0})
          if (acc > 0.77):
            saver.save(sess, 'mvnet1.ckpt')
            return
          t_x, t_y = get_xy(train_data[:10000])
          t_l, t_a = sess.run([cross_entropy, accuracy], feed_dict = {x:t_x, y:t_y, keep_prob:1.0})
        
          print "Step: %d \nLoss: %f, Accuracy: %f , Top_K accuracy: %f\nt_loss: %f, t_acc: %f"%(step, loss, acc, topk,t_l, t_a)
        
        step += 1
      if (write):
        saver.save(sess, 'mvnet1.ckpt')
  
  #train(True, True)






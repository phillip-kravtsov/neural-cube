from __future__ import print_function
from sys import argv
import tensorflow as tf
import numpy as np
import cube
import cube_nn as nn1
import cube_nn_c as nn2
import cube_nn_p2 as nn3
import handle_data as h_d
import time
import queue as Q
import copy
import math
import sys

script, choice = argv

q = cube.Cube(True)
F2L = False

def is_solved(scramble):
  if (F2L):
    if (not np.array_equal(scramble[1,:], np.ones(9,))):
      return False
    for i in range(2,6):
      if (not np.array_equal(scramble[i,3:], np.ones(6,) * i)):
        return False
    return True
  for i in range(6):
    solved = np.ones(9,) * i
    if (not np.array_equal(scramble[i,:], solved)):
      return False
  return True


moves = q.handscramble()

def expand(a):
  if (a < 2):
    return a
  else:
    return a + 4
def collapse(a):
  if (a < 2):
    return a
  else:
    return a - 4
def get_human(a):
  if (a < 0):
    return ""
  b =a% 6
  s = ""
  q = "UDFRBL"
  s += q[b]
  if (a < 6):
    return s
  if (a < 12):
    return (s + "2")
  else:
    return (s + "'")
for i in range(len(moves)):
    print(get_human(moves[i]))

total_time = 0.0
sum_time = 0.0
ct = 0
cth = 0
k = 0.43 # 1 / ln(10)
clip = 3.0
p1_sess = tf.Session(graph=nn1.graph)
p2_sess = tf.Session(graph=nn2.graph)
p3_sess = tf.Session(graph=nn3.graph)

nn1.saver.restore(p1_sess, "models/edgycubenn3.ckpt")
nn2.saver.restore(p2_sess, "models/mvnet.ckpt")
nn3.saver.restore(p3_sess, "models/cubennp2.ckpt")
root = [q.cube, [-1]]
mid_node = [q.cube[-1]]

def opp(q):
  q = q % 6
  if (q == 0):
    return 1
  if (q == 2):
    return 4
  if (q == 3):
    return 5
  return 6

def h(node):
  scramble = np.reshape(node[0], [1,-1])
  return math.floor(p1_sess.run(nn1.pred, feed_dict={
                    nn1.x:h_d.p1_h_process(scramble),
                    nn1.y: np.array([[1]]), nn1.keep_prob:1.0})) - 1.0

def p_h(scramble):
  #scramble is a cube array.
  scramble = np.reshape(scramble, [1,-1])
  return math.floor(p1_sess.run(nn1.pred, feed_dict={
                    nn1.x:h_d.p1_h_process(scramble),
                    nn1.y:np.array([[1]]), nn1.keep_prob:1.0})) - 1.0
def p_h2(scramble, prev_move):
  global ct
  global k
  ct += 1
  #t = time.time()
  scramble = np.reshape(scramble, [1,-1])
  #e = time.time()
  array = p2_sess.run(nn2.pred,
                      feed_dict={nn2.x:h_d.p2_m_process(scramble, prev_move),
                      nn2.y:np.zeros((1,10)),
                      nn2.keep_prob:1.0})
  array = k * -np.log(np.clip(array, 1e-10,1e3))
  #e = time.time()
  global sum_time
  #sum_time += (e - t)
  return array

def h2(node):
  global cth
  cth += 1
  scramble = np.reshape(node[0], [1,-1])
  t = time.time()
  h2 = p3_sess.run(nn3.pred, feed_dict={nn3.x:h_d.p2_h_process(scramble),
                                        nn3.y:np.array([[1]]), nn3.keep_prob:1.0})
  e = time.time()
  global sum_time
  sum_time += (e-t)
  f = math.floor(h2)
  return f - 1 if f > 1 else f

def is_goal(scramble, phase):
  for i in range(2):
    if (not np.all(scramble[i] < 2)):
      return False
  if (scramble[4,3] % 2 == 0  and scramble[4,5] % 2 == 0
      and scramble[2,3] % 2 == 0 and scramble[2,5]%2  == 0):
    return True
  else:
    return False

def ida_star_p1(root):
  #NODE: scramble, path from root
  #NODE: [][], [1, 17....]

  bound = h(root)

  for i in range (50):
    t = search_p1(root,0,bound)

    if (t == 999):
      mid_node[1] = [-1]
      ida_star_p2(mid_node)
      return

    bound = t
    print(i,  " ", bound)

def ida_star_p2(root):

  start_time = time.time()
  bound = 1
  for i in range(100):
    t = search_p2(root, 0, bound)

    if (t == 999):

      print(time.time() - start_time)
      return

    bound += 1
    #print "depth %d done. total time: %f h(p2) time: %f times h(p2) evaluated: %d "%(depth, time.time() - start_time, sum_time, ct)
    print("total time:%f h(p2) time: %f times h evaluated: %d times hp2 evaluated : %d "%(
            time.time()-start_time, sum_time, ct, cth))

def get_successors(node, phase):
  #TODO: split into phase 1 and 2
  p = node[1]
  prev_move = p[-1]
  ret = Q.PriorityQueue()
  array = np.copy(node[0])
  sarray = np.copy(array)
  if (phase == 2):
    estimates = p_h2(array, collapse(prev_move))

  for i in range(18):
    if (phase == 2 and (i % 6 > 1  and not ( i > 5 and  i < 14))):
      continue
    if (i % 6 == prev_move % 6):
      continue
    if (i % 6 == opp(prev_move % 6)):
      continue

    array =  q.execute_turn(i, array)
    path = copy.copy(node[1][:])
    path.append(i)
    if (phase == 1):
      estimate = p_h(array)
    else:
      estimate = estimates[0, collapse(i)]
    itime = time.time() - 1469861627.0
    ret.put(( estimate , itime, [np.copy(array), path]))
    array = np.copy(sarray)
  return ret

def search_p1(node, g, bound):

  h1 = h(node)
  f = g + h1
  if (h1 < -0.5):
    print(node[1])
  if (is_goal(node[0], 1)):
    for i in node[1]:
      print(get_human(i))
    global mid_node
    mid_node = node
    return 999
  if (f > bound):
    return f
  min = 100
  successors = get_successors(node, 1)
  while not successors.empty():
    succ = successors.get()
    t = search_p1(succ[2], g + 1, bound)
    if (t == 999):
      return t
    if (t < min):
      min = t
  return min

def search_p2(node, g, bound):

  h = h2(node)
  f = g + h
  if (h < 0.0):
    print(node[1])
  if (is_solved(node[0])):
    print("P2 Solved")
    print(node)
    for i in node[1]:
      print(get_human(i))
    return 999

  if (f > bound):
    return f
  if (len(node[1])> 20):
    return f

  minimum = 100
  successors = get_successors(node, 2)
  p = min(successors.qsize(), 7)

  for i in range(p):
    succ = successors.get()

    global clip
    t = search_p2(succ[2], g + min(succ[0], clip), bound)

    if (t == 999):
      return t
    if (t < minimum):
      minimum = t
  return minimum

if (choice == 'solve'):
    ida_star_p1(root)
else:
    print("Other options not yet supported.")

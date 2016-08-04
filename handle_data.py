import numpy as np


def cut_centers(data):
  data = np.delete(data, [4,13,22,31,40,49], 1)
  return data  
def get_data():
  global full_data
  return np.genfromtxt("data/data.txt", max_rows = 300000)
def cube_one_hot(cube):
  combined = cube == 0
  for i in range(1,6):
    combined = np.hstack([combined, cube == i])
  return combined.astype(np.int32)
def p1_mask(array):
  return array < 2
def eo(front, side):
  if (front < 2):
    return True
  if (front == 3 or front == 5):
    return False
  else:
    if (side < 2):
      return False
    else:
      return True
def p1_get_edges(cube):
  # given the cube, return an array of edge orientation
  e = []
  index = [0,1, 4,1, 0,3, 5,1, 0,5, 3,1, 0,7, 2,1, 4,5, 5,3, 4,3, 3,5, 2,3, 5,5, 2,5, 3,3, 1,1, 4,7, 1,3, 3,7, 1,5, 5,7, 1,7,2,7]
  #print len(index)
  for i in range(0, len(index), 4):
    front = (cube[index[i], index[i+1]])
    side = (cube[index[i+2], index[i+3]])
    e.append(eo(front, side))
  
  return np.array(e).astype(np.int32)

def one_hot(array, classes):
  ret = np.eye(classes)[array]
  #print expand(ret)
  return ret
def expand(array):
  return np.reshape(array, [1, array.shape[0]])

def p1_h_process(data):
  e = p1_get_edges(np.reshape(data, (6,-1)))
  #print e.shape
  e = np.reshape(e, [1,-1])
  data = cut_centers(data)
  #data = p1_mask(data)
  #print data.shape
  data = np.hstack([p1_mask(data), e])
  return data.astype(np.int32)

def p2_m_process(data, prev_move):
  data = cut_centers(data)
  data = cube_one_hot(data)
  #print data.shape
  data = np.hstack([data, expand(one_hot(prev_move, 10))])
  return data

def p2_h_process(data):
  data = cut_centers(data)
  data = cube_one_hot(data)
  return data
  
def process(full_data, with_labels, phase):
  if (full_data.ndim == 1):
    expand(full_data)
  print full_data.shape
  prev = False
  edges = []
  for i in full_data:
    edges.append(p1_get_edges(np.reshape(i[:-1], (6,-1))))
  e = np.array(edges)
  print e
  print e.shape
  full_data = cut_centers(full_data) 
  if (with_labels):
    if (prev):
      bin_inputs = full_data[:,:-11]
    else:
      bin_inputs = full_data[:,:-1]
  else:
    bin_inputs = full_data
  if (phase == 1):
    combined = np.hstack([bin_inputs < 2 , e])
    
  if (phase == 2):
    bin_inputs = bin_inputs[:,:]
    #print bin_inputs.shape
    w = bin_inputs == 0
    y = bin_inputs == 1
    g = bin_inputs == 2
    o = bin_inputs == 3
    b = bin_inputs == 4
    r = bin_inputs == 5
    combined = np.hstack([w,y,g,o,b,r])
  #print combined.shape
  
  if (with_labels):
    if(prev):
      labels = full_data[:,-11:]
    else:
      labels = np.reshape(full_data[:,-1] , (-1,1))

    combined = np.concatenate((combined,labels), axis=1)

  return combined.astype(np.int32)
def fix(labels):
  a = labels > 1
  labels[a] -= 4
  print labels
  return labels
def do_things():
  bin_data = process(get_data(), True, 1)
  f = open("data/p1_with_edges.txt", 'w')

  for i in range(bin_data.shape[0]):
    for j in range(bin_data.shape[1]):
      f.write(str(bin_data[i,j]))
      f.write(" ")
    f.write("\n")
def fix_labels():
  full_data = get_data()
  f = open("data/fdata2.txt", 'w')
  inputs = full_data[:,:-1]
  labels = full_data[:,-1]
  labels = fix(labels)
  labels = np.expand_dims(labels,axis=1)
  full_data = (np.concatenate((inputs, labels), axis=1)).astype(np.int32)
  for i in range(full_data.shape[0]):
    for j in range(full_data.shape[1]):
      f.write(str(full_data[i,j]))
      f.write(" ")
    f.write("\n")
#fix_labels()
#do_things()

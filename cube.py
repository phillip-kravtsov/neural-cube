import numpy as np

class Cube:
  def opp(self, x):
    if (x == 0):
      return 1
    if (x == 2):
      return 4
    if (x == 3):
      return 5
    else:
      return -1
  def handscramble_p2(self):
    ret = []
    while (len(ret) < 18):
      q = np.random.random_integers(0,17)
      if (len(ret) > 0):
        if (ret[-1] % 6 == q % 6):
          continue
        if (ret[-1] % 6 == self.opp(q%6) % 6):
          continue
      if (q % 6 > 1 and not (q > 5 and q < 14)):
        continue
      ret.append(q)
      self.execute_turn(q, self.cube)
    return ret
  def handscramble(self):
    ret = []
    while (len(ret) < 22):
      q = np.random.random_integers(0,17)
      if (len(ret) > 0):
        if (ret[-1] % 6  == q % 6):
          continue
        if (ret[-1] % 6 == self.opp(q % 6)):
          continue
      ret.append(q)
      self.execute_turn(q, self.cube)
    return ret
  def save(self):
    self.savec = np.copy(self.cube)

  def reset(self):
    self.cube = np.copy(self.savec)
  def __init__(self, initialize):
    self.cube = np.zeros([6,9])
    self.savec = np.zeros([6,9])
    if (initialize):
      for i in range(self.cube.shape[0]):
        self.cube[i,:] = i
        self.savec[i,:] = i
  def is_p1(self):
    for i in range(2):
      if (not np.all(self.cube[i] < 2)):
        return False
    return True
  def is_solved(self):
    for i in range(self.cube.shape[0]):
      if (not np.array_equal(self.cube[i], (np.ones(9,) * i))):
        return False
    return True     
  def execute_turn(self, turn, cube):
    d = np.copy(cube)
    if (turn < 0):
       return cube
    if (turn >= 0):
      if (turn % 6 == 0):
        cube[2,0] = d[3,0]
        cube[2,1] = d[3,1]
        cube[2,2] = d[3,2]
        
        cube[3,0] = d[4,0]
        cube[3,1] = d[4,1]
        cube[3,2] = d[4,2]

        cube[4,0] = d[5,0]
        cube[4,1] = d[5,1]
        cube[4,2] = d[5,2]

        cube[5,0] = d[2,0]
        cube[5,1] = d[2,1]
        cube[5,2] = d[2,2]
      
      elif (turn % 6 == 1):
        cube[2,6] = d[5,6]
        cube[2,7] = d[5,7]
        cube[2,8] = d[5,8]
        
        cube[5,6] = d[4,6]
        cube[5,7] = d[4,7]
        cube[5,8] = d[4,8]
        
        cube[4,6] = d[3,6]
        cube[4,7] = d[3,7]
        cube[4,8] = d[3,8]
        
        cube[3,6] = d[2,6]
        cube[3,7] = d[2,7]
        cube[3,8] = d[2,8]
      
      elif (turn % 6 == 2):
        cube[0,6] = d[5,8]
        cube[0,7] = d[5,5]
        cube[0,8] = d[5,2]

        cube[3,0] = d[0,6]
        cube[3,3] = d[0,7]
        cube[3,6] = d[0,8]
        
        cube[1,6] = d[3,0]
        cube[1,7] = d[3,3]
        cube[1,8] = d[3,6]
        
        cube[5,8] = d[1,6]
        cube[5,5] = d[1,7]
        cube[5,2] = d[1,8]
        
      elif (turn % 6 == 3):
        cube[0,2] = d[2,2]
        cube[0,5] = d[2,5]
        cube[0,8] = d[2,8]

        cube[4,6] = d[0,2]
        cube[4,3] = d[0,5]
        cube[4,0] = d[0,8]

        cube[1,6] = d[4,6]
        cube[1,3] = d[4,3]
        cube[1,0] = d[4,0]
        
        cube[2,2] = d[1,6]
        cube[2,5] = d[1,3]
        cube[2,8] = d[1,0]

      elif (turn % 6 == 4):
        cube[0,0] = d[3,2]
        cube[0,1] = d[3,5]
        cube[0,2] = d[3,8]
        
        cube[3,2] = d[1,0]
        cube[3,5] = d[1,1]
        cube[3,8] = d[1,2]

        cube[1,0] = d[5,6]
        cube[1,1] = d[5,3]
        cube[1,2] = d[5,0]
        
        cube[5,6] = d[0,0]
        cube[5,3] = d[0,1]
        cube[5,0] = d[0,2]

      elif (turn % 6 == 5):
        cube[0,0] = d[4,8]
        cube[0,3] = d[4,5]
        cube[0,6] = d[4,2]

        cube[4,2] = d[1,2]
        cube[4,5] = d[1,5]
        cube[4,8] = d[1,8]
        
        cube[1,2] = d[2,6]
        cube[1,5] = d[2,3]
        cube[1,8] = d[2,0]

        cube[2,0] = d[0,0]
        cube[2,3] = d[0,3]
        cube[2,6] = d[0,6]

      s = turn % 6
      cube[s,0] = d[s,6]
      cube[s,1] = d[s,3]
      cube[s,2] = d[s,0]
      cube[s,6] = d[s,8]
      cube[s,7] = d[s,5]
      cube[s,8] = d[s,2]
      cube[s,3] = d[s,7]
      cube[s,5] = d[s,1]

      turn -= 6
      return self.execute_turn(turn, cube)




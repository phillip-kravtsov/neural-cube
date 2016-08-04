import kociemba
import cube
import numpy as np
#'DRLU U BFBR
#BLUR R LRUB
#LRDD F DLFU
#FUFF D BRDU
#BRUF L LFDD
#BFLU B LRBD'

def get_int(a):
    d = a.split()
    string = "UDFRBL"
    ret = []
    for i in xrange(len(d)):
        base = string.find(d[i][0]) 
        
        if (len(d[i]) == 1):
            ret.append(base)
        elif(d[i][1] == "2"):
            ret.append(base + 6)
        elif(d[i][1] == "'"):
            ret.append(base + 12)
    return ret

def get_human(a):
    ret = ""
    #print a
    for i in range(len(a)):
        if (a[i] < 0):
            continue
        b = a[i] % 6
        s = ""
        q = "UDFRBL"
        s += q[b]
        if (a[i] < 6):
            ret += s
        elif (a[i] < 12):
            s += "2"
            ret += s
        else:
            s += "'"
            ret += s
        ret += " "
    return ret
def to_lc(i):
    #to letters, by character
    s = "UDFRBL"
    return s[int(i)]
def to_lfd(d):
    #fix d --> rearrange
    #current :UDRB D FFUF 
    #goal:    FUFF D BRDU
    ret = ""
    for i in xrange(len(d)):
        ret = d[i] + ret
    return ret

def to_l(cube):
    cube = np.array([cube[0,:], cube[3,:],cube[2,:],cube[1,:],cube[5,:],cube[4,:]])
    #print cube.shape

    cube = np.reshape(cube, [-1,])
    #print cube.shape 
    b_fix = ""
    for i in xrange(cube.shape[0]):
        b_fix += to_lc(cube[i])
    ret = b_fix[:27]
    ret += to_lfd(b_fix[27:36])
    ret += b_fix[36:]
    #print len(ret)
    return ret

def get_len(soltn):
    extra = "'2 "
    ct = 0
    for i in xrange(len(soltn)):
      if (soltn[i] not in extra):
        ct+= 1
    return ct

def split_phases(slist):
    #Splits the soltn list into phase 1 and 2.
    #Find the last P1 move (2-5, 14-17)
    index = 0
    for i in range (len(slist)-1, -1, -1):
        if (slist[i] % 6 > 1 and (slist[i] < 6 or slist[i] > 13)):
            index = i
            break
    return slist[:i+1] , slist[i+1:]
def collapse(move):
  if (move < 2):
    return move
  else:
    return move - 4
def init_data():
  q = cube.Cube(True)
  f = open("data/p2data.txt", 'w')
def write(f, cube):
    cube = cube.astype(np.int32)
    
    for i in range(cube.shape[0]):
        for j in range(cube.shape[1]):
            f.write(str(cube[i,j]))
            f.write(" ")
def get_distance(cube):
  soltn = kociemba.solve(to_l(cube))
  return get_len(soltn)
  
def mwrite(f, m):
  if (m is None):
    for i in range(10):
      f.write("0 ")
  else:
    one_hot = np.eye(10)[collapse(m)].astype(np.int32)
    for i in one_hot:
      f.write(str(i))
      f.write(" ")

def gen():
  for i in range(19000):
    if (i % 100 == 0):
        print i
    q.handscramble_p2()
    c = np.copy(q.cube)
    soltn = kociemba.solve(to_l(q.cube))
    solution = get_int(soltn)
    #print solution
    p1, p2 =  split_phases(solution)
    for j in range(len(p1)):
        c = q.execute_turn(p1[j] , c)
    write(f,c)
    f.write(str(len(p2)))
    f.write("\n")
    #print p1, "   ", p2
    for j in range(len(p2)):
        c = q.execute_turn(p2[j], c)
        write(f,c)
        f.write(str(len(p2) - j - 1))
        f.write("\n") 
    #record cube, len(p2)
    #apply turn, record len(p2 - 1)...
    #end
def gen_mv():
 
  for i in range(50000):
    if (i% 100 == 0):
      print i
    global q
    q.handscramble_p2()
    c = np.copy(q.cube)
    soltn = kociemba.solve(to_l(c))
    solution = get_int(soltn)
    p1, p2 = split_phases(solution)
    #print len(p2)
    for j in range(len(p1)):
      c = q.execute_turn(p1[j], c)
    #print c
    for j in range(len(p2)):
      write(f, c)
      if (j > 0):
        mwrite(f, p2[j-1])
      else:
        mwrite(f, None)

      f.write(str(p2[j]))
      f.write("\n")
      
      c = q.execute_turn(p2[j], c)
#gen()    



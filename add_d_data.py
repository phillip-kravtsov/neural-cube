
d = open("data/f_a_cases.txt", 'r')
a = open("data/p2_h_data.txt", 'a')

for line in d:
  a.write(line)


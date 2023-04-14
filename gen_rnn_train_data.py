import os
import sys

data = open('./res_all_20230320.txt', 'r')

for line in data:
    a = line[-3]
    # f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms"
    fp = open(f"./RNN_Train_in/{a}.in", 'a')
    fp.write(line[1:-5] + '\n')
    # print(line[1:-5] + '\n')
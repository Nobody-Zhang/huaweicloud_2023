import os
import time
file = r'D:\Workspace\github\yolov5\train-data'
for root, dirs, files in os.walk(file):
    for file in files:
        path = os.path.join(root, file)
        t1 = time.time()
        print(path) #当前的绝对路径
        os.system("python mixed.py --weights best1.pt --source %s --nosave" % path)
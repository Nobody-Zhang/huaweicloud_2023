import cv2
import time
from Nanodet import NanoDet

a = NanoDet("seg_face/seg_face.xml", 4)
video = cv2.VideoCapture("day_man_001_20_2.mp4")
frame = 0

total_time = 0

while True:
    success, im = video.read()
    frame += 1
    if not success:
        break
    print(f'frame {frame}:', end=" ")
    t1 = time.time()
    b = a.detect(im, 0.6, 0.8)
    total_time += time.time() - t1

print(total_time / frame)
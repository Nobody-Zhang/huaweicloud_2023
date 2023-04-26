import cv2
import time
# import model wrapper class
from openvino.model_zoo.model_api.models import NanoDetPlus
# import inference adapter and helper for runtime setup
from openvino.model_zoo.model_api.adapters import OpenvinoAdapter, create_core

video = cv2.VideoCapture("day_man_001_10_1.mp4")

# define the path to mobilenet-ssd model in IR format
model_path = "seg_face\seg_face.xml"


# create adapter for OpenVINO™ runtime, pass the model path
model_adapter = OpenvinoAdapter(create_core(), model_path, device="CPU")

# create model API wrapper for SSD architecture
# preload=True loads the model on CPU inside the adapter
nanodet_model = NanoDetPlus(model_adapter, configuration={'num_classes':4}, preload=True)

total_time = 0
frame = 0

while True:
    success, im = video.read()
    #im = cv2.resize(im, (416,416))
    frame += 1
    if not success:
        break
    print(f'frame {frame}:', end=" ")
    t1 = time.time()
    b = nanodet_model(im)
    """
    for detection in b[0]:
        
        这儿的对象是如下的类型
        <openvino.model_zoo.model_api.models.utils.Detection object at 0x7fa312c7c700>
        可以参考https://github.com/openvinotoolkit/open_model_zoo/blob/d91af68779698e2b6616898ca78263bd02e35a7d/demos/common/python/openvino/model_zoo/model_api/models/utils.py#L22
        里面的Detection 类
        
        print(detection.xmin)
        print(detection.ymin)
        print(detection.xmax)
        print(detection.ymax) # bounding boxes
        print(detection.id) #这里需要转化一下
        print(detection.score) # 正确率
        if(detection.id == 0):
            dt = im[detection.ymin: detection.ymax, detection.xmin: detection.xmax]
            cv2.imshow("cropped", dt)
            cv2.waitKey(1)
    """
    total_time += time.time() - t1

print(total_time / frame)


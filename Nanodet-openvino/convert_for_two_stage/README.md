# FOLLOW THESE STEPS
1. 将main.cpp和nanodet.cpp中的库文件替换为nanodet_openvino_*.h
2. 替换main.cpp中的（164行）class_names
3. 替换main.cpp中的如下地方
```cpp
336 auto detector = NanoDet("nanodet.xml"); 
```

```cpp
336 auto detector = NanoDet("*****.xml"); 
```

#### 应该就可以了

## ~~错了不怪我~~

# UPDATE FOR PYTHON

第一个阶段切分人脸的时候，用当前目录下的seg_face文件夹中的seg_face.xml(应该是)去初始化Nanodet_openvino。

第二个阶段人脸切眼睛和嘴巴的时候，用face_eyes文件夹中的mouth_eyes.xml(应该是)去初始化

每个文件夹下的.bin .xml .mapping是推理的时候会用到的文件，其他的是转换中间产物~~~


# UPDATE FOR OPENVINO_TOOLKIT(OFFICIAL)
想要用openvino官方的工具转换，可以参考这个仓库：

https://github.com/openvinotoolkit/open_model_zoo/blob/d91af68779698e2b6616898ca78263bd02e35a7d/demos/common/python/openvino/model_zoo/model_api/README.md


下面是部署的流程：
1.  clone openvino Model Zoo的仓库：
```bash
git clone https://github.com/openvinotoolkit/open_model_zoo.git
```
2.  部署API
```bash
pip install <omz_dir>/demos/common/python
```
其中omz_dir是open_model_zoo的路径
部署完成之后，
尝试运行
```bash
python -c "from openvino.model_zoo import model_api"
```
如果没有报错，说明部署成功

3.  转换模型（可以跳过，我们自己已经转换好了模型--->关键的是.xml .mapping .bin文件，需要放到同一个文件夹之下）
```bash
python3 <omz_dir>/tools/downloader/converter.py --name <model_name>
```

4. 直接使用openvino的API进行推理
```python
import cv2
import time
# import model wrapper class
from openvino.model_zoo.model_api.models import NanoDetPlus
# import inference adapter and helper for runtime setup
from openvino.model_zoo.model_api.adapters import OpenvinoAdapter, create_core

video = cv2.VideoCapture("night_woman_005_31_4.mp4")

# define the path to mobilenet-ssd model in IR format
model_path = "/home/hzkd/nanodet/python_demo/convert_for_two_stage/nanodet.xml"

# create adapter for OpenVINO™ runtime, pass the model path
model_adapter = OpenvinoAdapter(create_core(), model_path, device="CPU")

# create model API wrapper for SSD architecture
# preload=True loads the model on CPU inside the adapter
# nanodet_model = NanoDetPlus(model_adapter, preload=True)
# Update: *** openvino文档写的跟***一样
nanodet_model = NanoDetPlus(model_adapter, configuration={'num_classes':4}, preload=True)
# configuration={'num_classes':4} 这个字典来自定义一些传入的参数，比如num_classes，可以看源码（


total_time = 0
frame = 0

while True:
    success, im = video.read()
    frame += 1
    if not success:
        break
    print(f'frame {frame}:', end=" ")
    t1 = time.time()
    b = nanodet_model(im)
    for detection in b[0]:
        """
            这儿的对象是如下的类型
            <openvino.model_zoo.model_api.models.utils.Detection object at 0x7fa312c7c700>
            可以参考https://github.com/openvinotoolkit/open_model_zoo/blob/d91af68779698e2b6616898ca78263bd02e35a7d/demos/common/python/openvino/model_zoo/model_api/models/utils.py#L22
            里面的Detection 类
        """
        print(detection.xmin)
        print(detection.ymin)
        print(detection.xmax)
        print(detection.ymax) # bounding boxes
        print(detection.id) #这里需要转化一下
        print(detection.score) # 正确率

    total_time += time.time() - t1

print(total_time / frame)

```
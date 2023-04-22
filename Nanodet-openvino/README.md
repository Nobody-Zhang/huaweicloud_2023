## 更新Python版Nanodet_Openvino
文件包为Nanodet.py

### 使用
```python
from Nanodet import NanoDet

model = NanoDet(model_path=..., num_class=...)
img = cv2.imread(...)
boxes = model.detect(img)
```

boxes即为检测到的物体的Bbox
其中每个box包含
```
BoxInfo:
	x1
	y1
	x2
	y2
	score
	label
```

### 注释写得挺详细的，有问题看注释

### ~~有点bug, 正在修，先看看~~

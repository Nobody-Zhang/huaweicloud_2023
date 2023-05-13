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
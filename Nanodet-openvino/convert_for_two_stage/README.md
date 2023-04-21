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

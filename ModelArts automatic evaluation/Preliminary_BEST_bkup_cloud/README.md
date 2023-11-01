# 生产队的大萝卜 初赛刷榜代码(BEST SCORE: 0.9741)

## 简介

本项目是一个基于 YOLO（You Only Look Once）以及openvino的目标检测系统。我们的主要代码位于 `yolo/yolo.py` 文件中。由于时间限制，我们没有将主要代码嵌入到 `customize_service.py` 文件中。因此，在 `customize_service.py` 文件的 `__init__` 方法中没有初始化模型。

**模型的初始化代码位于 `yolo.py` 文件中的 "Init model" 段落中。因此，我们的计时从 "Run inference" 开始，一直到整个函数结束。**

## 主要文件结构

```diff
.
│  config.json
│  customize_service.py
│  openmodelzoo_modelapi-2022.3.0-py3-none-any.whl
│  README.md
│  scikit_image-0.19.3-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl
│  scikit_learn-1.0.2-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
└─yolo
    │  export.py
    │  yolo.py
    ├─models
    ├─utils
    └─yolov5s_best_openvino_model_supple_quantization_FP16
            best.bin
            best.mapping
            best.xml
```

## 使用方法（参考customize_service.py）

1. 首先，确保已经安装了项目的依赖项。
2. 在需要进行目标检测的地方，调用 `yolo/yolo.py` 中的 `yolo_run` 方法。
3. 传入参数格式为`source = 'finename'`。
4. 输出参数格式如下，其中，category为驾驶行为类别编号，duration为算法推理耗时(ms)

```json
{
  "result":
  {
    "category":0,
    "duration":6000
  }
}
```

## 测试说明

部署为AI应用时，采用以下策略:

类型：**动态加载+自定义算法**

运行环境：**pytorch_1.8.0-cuda_10.2-py_3.7-ubuntu_18.04-x86_64**

AI引擎：**PyTorch**

已部署成功ID:

```
e01ba0f9-9670-4d9f-bc2b-e5fbb76b3ca3
```
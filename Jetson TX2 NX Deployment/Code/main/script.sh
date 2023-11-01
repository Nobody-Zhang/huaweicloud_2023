#!/bin/bash

# 设置 LD_PRELOAD 环境变量
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

# 清除 GStreamer 缓存
rm ~/.cache/gstreamer-1.0/*

# 运行 Python 脚本，并传递两个参数
python3 "$1" /home/jetson/Downloads/deepstream/deepstream-6.0/samples/streams/AnyConv.com__night_woman_063_30_1.h264 


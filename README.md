# 第十八届“挑战杯”全国大学生课外学术科技作品竞赛“揭榜挂帅”专项赛 · 华为云赛道 国二

## [赛事首页](https://competition.huaweicloud.com/information/1000041855/introduction)

项目思路：

云端刷榜算法利用 yolov5 + Openvino + Huawei Cloud ModelArts 实现，
其中采用了二分以及分治的算法优化，使得不用遍历整个视频即可得到疲劳状态的起始、终止时间点。

端侧 ( Jetson TX2 NX ) 的算法是用的 Deepstream ( C / C++ 6.0.1 ) + yolov5 . 其中有 

On-the-Air Model Update [在线模型更新](https://docs.nvidia.com/metropolis/deepstream/6.0.1/dev-guide/text/DS_on_the_fly_model.html)

Smart Video Record [智能视频记录](https://docs.nvidia.com/metropolis/deepstream/6.0.1/dev-guide/text/DS_Smart_video.html)

的具体实现代码（C/C++ ,目前Python的实现NVIDIA官方说的无法实现orz）


具体实现参见本目录下的技术文档~

```
Team Name: 生产队的大萝卜
Program Leader: 张恭博
Directors: 周健 吴非
Teammates: 郭树明 吕露然 张奥琳 陈星宇 吴锦添 贾雨凡 周喆宇 张家豪 张锦深

Acknowledge: 唐岷瀚 赖永烨 邓皓宇 张诗语
```


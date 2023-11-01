## MTCNN-Landmarks

* 目录结构

|-- detect.py：用于检测视频或者照片，命令行参数见文件中参数解析

|-- inferAPI.py：封装了对图片以及视频的推理，输入为路径，设备等

|-- infer_examples.py：调用接口示例

|-- infer_models：保存模型的目录

|-- models：网络模型代码

|-- test：infer_examples.py中的测试案例

|--utils：工具类包

* 环境依赖

与原环境一致，唯一多需要安装一个torchsummary包

* 运行

运行案例

~~~bash
python infer_example.py
~~~

运行可视化检测

~~~bash
python detect.py --model_path ./infer_models --images_path ./test --save_path ./output --device 0
~~~










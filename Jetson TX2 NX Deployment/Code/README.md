# 生产队的大萝卜 倦旅智鉴

由于模型较大，我们将后续上传至OTA以供查看

## dist: 前端

点开html即可

设计了UI的界面，如动态背景，关于界面，按钮设计，通路设计，以及相应的代码的实现。

编写了vue的代码，实现了动态的能够与后端进行实时的数据交换。

## ota_test: 后端

Makefile配置CUDA版本，并且配置好Deepstream==6.0.1、opencv、gst、gst-rtsp-server即可运行，

注意Makefile中的Deepstream路径可能跟您的路径不一样，需要更换

camera_input_rtsp_output.py可以快速实现部分功能

总代码在main.cpp之下

```shell
sudo make
./deepstream-customized
```

send_int.py以及trysend.py代码建立前后端连接

MP4_input.py以mp4作为输入



## mtcnn_landmarks：mtcnn

云侧推理代码。



## 在线模型微调

分为云侧部分以及端侧部分

端侧部分代码于**APIGW-python-sdk-2.0.4**中，云侧代码于**Cloud_finetune**中。

不用部署，本地代码会自动用这个代码在云侧部署训练任务。



## WatchDog: 监视进程

安装过程如下

```shell
mkdir build
cd build
cmake ..
make
```



## VoiceInteraction: 语音/大语言模型部分使用说明


### 目录结构

本目录下总共有三个文件夹，分别是audio_bags,用来在离线状态下播放预先录制好的语音包；scripts，放置和VoiceInteraction相关的代码（python）；
temp_audio,用来在进行语音回复的时候用来暂时存放从华为云端接收到的音频文件并播放（建议在每次运行完后都清空这个文件夹）。

### VoiceInteraction类的使用

在类VoiceInteraction中封装了关于语音交互的相关函数，其中包括通过调用华为云进行语音识别、调用学校服务器部署模型llama对给定文本进行文本生成、
调用华为云服务进行音频合成三个服务型内部函数，一般不需要在外部调用。针对外部调用的函数主要是`alert(self, audio)`函数和`communicate(self)`函数。

#### 针对外部调用函数的说明

`alert(self, status)`函数是在当收到websocket发送来的驾驶员的异常状态的时候调用该函数，需要传入参数audio,这里的audio既可以是一个本地的"*.wav"音频文件，
也可以是字节流形式的音频数据，该函数接下来将通过语言模型生成对应的文本来警告驾驶员不能以危险状态进行驾驶，然后通过华为云服务生成对应音频并播放。

`communicate(self)`函数是与用户进行语音交互并且接收特定用户指令的函数。该函数在调用的外层一般是一个死循环，只有当识别到用户说了“再见”以后才会跳出循环。
其它关键字还有“语音助手”，语句中含有这个关键字将会触发文本生成函数，并调用华为云服务生成音频并播放回答；“记录”，语句中包含该关键字将会通过websocket
发送命令并开始进行*smart video record*。

#### 类内其它函数的说明

`close_websocket(self,client)`函数，专门用来关闭websocket的函数，在函数`SASR(self,audio)`中调用，专门开一个线程出去关闭websocket，
这样主线程就不会因为关闭websocket而被阻塞。websocket关闭之后该线程将自动被消去。

`SASR(self, audio, audio_format,property)`函数，通过调用华为云服务来进行语音识别。其中，audio参数可以是一个本地的"*.wav"文件，也可以是一段
字节流形式的音频，该函数中将自动对audio的类型进行判断。如果华为云进行语音识别的置信度>0.5,那么就返回华为云语音识别的结果，为字符串形式的文本，
否则的话将返回None。如果华为云语音识别失败的话也将返回None。`audio_format`参数和`property`参数都是带有默认值的，代表音频类型和将要识别的音频中文字的种类。
根据我们当前的项目，我们给他们分别赋默认值"pcm16k16bit"和"chinese_16k_general"。

`llama(self,input_question,max_tokens,temperature,num_beams,top_k)`函数，是通过调用学校服务器上的服务进行实时文本生成的函数。
参数`input_quesiont`是str类型的字符串，其它参数都带有默认值，是要求和文本生成时候相关的参数，具体意义为：
1.max_tokens：新生成的句子的token长度。
2.temperature：在0和2之间选择的采样温度。较高的值如0.8会使输出更加随机，而较低的值如0.2则会使其输出更具有确定性。temperature越高，使用随机采样最为decoding的概率越大。
3.num_beams:当搜索策略为束搜索（beam search）时，该参数为在束搜索（beam search）中所使用的束个数，当num_beams=1时，实际上就是贪心搜索（greedy decoding）。
4.top_k：在随机采样（random sampling）时，前top_k高概率的token将作为候选token被随机采样。
具体使用requests模块向学校服务器发送请求，并获得对应的json。对json进行解析，就可以获得对应生成的文本。

`TTCS(self,given_tent,property,audio_format,sample_rate,volume,pitch,speed)`函数，通过调用华为云服务来进行语音合成。
其中，`given_text`是传入的将要生成音频文件的文本，其它参数都带有默认值，是和要生成的音频相关的参数设置。
1.audio_format：待合成的音频格式，可选mp3，wav等，默认wav。
2.pitch：音高，[-500,500] ，默认是0。
3.speed：语速，[-500,500] ，默认是0。
4.volume：音量，[0,100]，默认是50。
5.sample_rate：采样率，支持“8000”、“16000”，默认“8000”。

`asked(self,audio)`函数，`是communicate(self,audio)`函数中将要调用的函数，对于音频中不同的关键字将进行不同的操作，包括”记录“，”再见“，
“语音助手”等，还包括当华为云识别失败的时候播放的默认语音。

### warning2.py的具体使用

首先导入相关的包，接下来实例化`VoiceInteraction`类的对象`vi`，然后准备websocket相关需要在最开始运行的代码，与特定端口进行联系，
如果连接失败则阻塞在此处，直到连接成功。

接下来，针对三个线程定义三个不同的函数。函数`receive()`专门用来通过websocket接收驾驶员的异常状态信息，并将一床状态信息赋值给全局变量`warning_type`，
`communicate()`在一个死循环内调用`vi.communicate()`方法。同时通过线程锁，防止该线程和其它线程同时通过音响播放音频，造成混乱。
`handle_warning()`函数，把全局变量`warning_type`赋值给`cur_type`,然后判断`cur_type`和`last_type`是否相等，如果相等的话就不调用`vi.alert()`方法，
否则就是在重复播放音频，如果不相等且`cur_type`不等于0，说明是一个和之前的状态都不一样的异常状态，那么就需要调用`vi.alert()`方法提醒驾驶员。

最后通过`thread`开三个线程分别运行三个函数。
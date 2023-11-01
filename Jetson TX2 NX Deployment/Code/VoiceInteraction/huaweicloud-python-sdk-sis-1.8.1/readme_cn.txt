安装：
	1. 请确认已安装Python包管理工具setuptools，请确认已安装requests和websocket-client，可通过“pip list”命令查看已安装列表。如果没有安装，请使用以下命令安装 
		pip install setuptools
		pip install requests
		pip install websocket-client
	2. 进入sdk的目录，执行以下安装命令：
		python setup.py install

使用步骤：
	1. “cn_demo”文件夹：存放中国站上线接口调用示例代码，代码文件和接口对应关系如下：
		一句话识别Http接口        ： sasr_demo.py
		一句话识别Websocket接口   ： sasr_websocket_demo.py
		录音文件识别              :  lasr_demo.py
		录音文件极速版			  ： flash_lasr_demo.py
		实时语音转写              ： rasr_demo.py
		语音合成                  ： tts_demo.py
		热词使用				  ： hot_word_demo.py
		口语评测 & 多模态评测	  ： pa_demo.py
		实时语音合成			  ： rtts_demo.py
	2. “intl_demo”文件夹：存放国际站上线接口调用示例代码，代码文件和接口对应关系如下：
		一句话识别 ： asr_customization_demo.py
	3. “data”文件夹：存放示例音频可供参考使用。

注意：
	1. python sdk目前仅支持python3，暂不支持python2
	2. 中国站使用中国站账户登录，仅支持识别中文；国际站使用国际站账号登录，仅支持识别英文。
	3. 国际站一句话识别暂时不支持热词功能，不支持digit_norm属性。
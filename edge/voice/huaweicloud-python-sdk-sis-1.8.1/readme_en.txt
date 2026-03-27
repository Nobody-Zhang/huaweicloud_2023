intall:
	1. Please make sure you have installed "setuptools", "requests" and "websocket-client"，you can run "pip list" to see the installed packages, and you can run the following commands to install packages.
		pip install setuptools
		pip install requests
		pip install websocket-client
	2. Go to the sdk directory and run the command：
		python setup.py install

usage:
	1. In folder "cn_demo", the code show how to use the SDK in chinese region.
	2. In folder "intl_demo", the code show how to use the SDK in international region.
	3. In folder "data", the audio files are examples for users. 
	
warn：
	1. The SDK can only be runned in python3, and the sdk is incompatible with python2.
	2. In chinese region, the api only support recognizing chinese. In international region, the api only support recognzing english.
	3. In international region, sentence transcription doesn't support hot word or digit_norm.
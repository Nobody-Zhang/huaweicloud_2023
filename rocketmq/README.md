# rocketmq-client-python

## Prerequisites
RocketMQ 需要 JAVA 环境，执行 java -version 检查java版本：
```bash
$ java -version
java version "1.8.0_121"
```

安装RocketMQ：
```bash
# Download release from the Apache mirror
$ wget https://dist.apache.org/repos/dist/release/rocketmq/5.1.1/rocketmq-all-5.1.1-bin-release.zip

# Unpack the release
$ unzip rocketmq-all-5.1.1-bin-release.zip
```

使用Rocketmq-client-python需要cpp的底层库：
```bash
wget https://github.com/apache/rocketmq-client-cpp/releases/download/2.0.0/rocketmq-client-cpp-2.0.0.amd64.deb

sudo dpkg -i rocketmq-client-cpp-2.0.0.amd64.deb

pip install rocketmq-client-python
```

执行setup.py:
```bash
python setup.py install
```

## Quick Start

运行 consumer.py 和 producer.py 需要启动 mqnamesrv 和 mqbroker：
```bash
cd rocketmq-all-5.1.1-bin-release

### start Name Server 
$ nohup sh bin/mqnamesrv &

### check whether Name Server is successfully started
$ tail -f ~/logs/rocketmqlogs/namesrv.log
The Name Server boot success...

### start Broker
$ nohup sh bin/mqbroker -n localhost:9876 &

### check whether Broker is successfully started, eg: Broker's IP is 192.168.1.2, Broker's name is broker-a
$ tail -f ~/logs/rocketmqlogs/broker.log
The broker[broker-a, 192.169.1.2:10911] boot success...

$ cd rocketmq-python
$ python src/producer.py
$ python src/consumer.py

```

## Usage

### Producer

```python
from rocketmq.client import Producer, Message

producer = Producer('PID-XXX')
producer.set_name_server_address('127.0.0.1:9876')
producer.start()

msg = Message('YOUR-TOPIC')
msg.set_keys('XXX')
msg.set_tags('XXX')
msg.set_body('XXXX')
ret = producer.send_sync(msg)
print(ret.status, ret.msg_id, ret.offset)
producer.shutdown()
```

### PushConsumer

```python
import time

from rocketmq.client import PushConsumer, ConsumeStatus


def callback(msg):
    print(msg.id, msg.body)
    return ConsumeStatus.CONSUME_SUCCESS


consumer = PushConsumer('CID_XXX')
consumer.set_name_server_address('127.0.0.1:9876')
consumer.subscribe('YOUR-TOPIC', callback)
consumer.start()

while True:
    time.sleep(3600)

consumer.shutdown()

```

## License
[Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0.html) Copyright (C) Apache Software Foundation

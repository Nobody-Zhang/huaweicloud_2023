# -*- coding: utf-8 -*-

from rocketmq.client import PushConsumer, ConsumeStatus
import time
import cv2
import base64
from PIL import Image
from io import BytesIO
import numpy as np



# 读取编码后的图像字符串并转回图像
def str2img(img_str: str):
	# base64 解码图像字符串
    img_bs64 = base64.b64decode(img_str)
    # 使用 pillow Image读取byte流
    #(这里我试过用np 和 cv2 都不好用，用cv2会丢失维度信息)
    pil_img = Image.open(BytesIO(img_bs64))
    # 转为array 格式
    img_rgb = np.asarray(pil_img)
    # 把bgr 转为 rgb
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    return img_bgr

def callback(msg):
    # print(msg.id, msg.body,  msg.get_property('property'))
    img = str2img(msg.body)
    cv2.imwrite('./2.jpg',img)
    return ConsumeStatus.CONSUME_SUCCESS


def start_consume_message():
    consumer = PushConsumer('consumer_group')
    consumer.set_name_server_address('127.0.0.1:9876')
    consumer.subscribe('TopicTest', callback)
    print ('start consume message')
    consumer.start()

    while True:
        time.sleep(3600)

if __name__ == '__main__':
    start_consume_message()
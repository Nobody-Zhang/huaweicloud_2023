# -*- coding: utf-8 -*-

from rocketmq.client import Producer, Message, TransactionMQProducer, TransactionStatus
import os
import sys
from pathlib import Path
import time
import threading
from threading import Thread
import cv2
import base64
from io import BytesIO

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

topic = 'TopicTest'
gid = 'test'
name_srv = '127.0.0.1:9876'
MUTEX = threading.Lock()

def img2str(img_path:str):
    img = cv2.imread(img_path)
    img_data = cv2.imencode('.jpg', img)[1].tobytes()  
    base64_data = base64.b64encode(img_data)
    img_base64 = str(base64_data, encoding='utf-8')
    return img_base64


def create_message():
    msg = Message(topic)
    msg.set_keys('XXX')
    msg.set_tags('XXX')
    msg.set_property('property', 'test')
    msg.set_body('message body')
    return msg

def create_message1():
    img_path = str(ROOT / '1.jpg')
    imgdata = img2str(img_path)
    msg = Message(topic)
    msg.set_keys('XXX')
    msg.set_tags('XXX')
    msg.set_property('property', 'test')
    # msg.set_body('another message body')
    msg.set_body(imgdata) 
    return msg

def send_message_sync(count):
    producer = Producer(gid)
    producer.set_name_server_address(name_srv)
    # producer.set_namesrv_addr(name_srv)
    producer.start()
    for n in range(count):
        msg = create_message()
        ret = producer.send_sync(msg)
        print ('send message status: ' + str(ret.status) + ' msgId: ' + ret.msg_id)
    print ('send sync message done')
    producer.shutdown()


def send_message_multi_threaded(retry_time):
    producer = Producer(group_id = gid , max_message_size = 4 * 1024 * 1024)
    producer.set_name_server_address(name_srv)
    msg = create_message()

    global MUTEX
    MUTEX.acquire()
    try:
        producer.start()
    except Exception as e:
        print('ProducerStartFailed:', e)
        MUTEX.release()
        return

    try:
        for i in range(retry_time):
            ret = producer.send_sync(msg)
            if ret.status == 0:
                print('send message status: ' + str(ret.status) + ' msgId: ' + ret.msg_id)
                break
            else:
                print('send message to MQ failed.')
            if i == (retry_time - 1):
                print('send message to MQ failed after retries.')
    except Exception as e:
        print('ProducerSendSyncFailed:', e)
    finally:
        producer.shutdown()
        MUTEX.release()
        return


def send_message_multi_threaded1(retry_time):
    producer = Producer(gid)
    producer.set_name_server_address(name_srv)
    msg = create_message1()

    global MUTEX
    MUTEX.acquire()
    try:
        producer.start()
    except Exception as e:
        print('ProducerStartFailed:', e)
        MUTEX.release()
        return

    try:
        for i in range(retry_time):
            ret = producer.send_sync(msg)
            if ret.status == 0:
                print('send message status: ' + str(ret.status) + ' msgId: ' + ret.msg_id)
                break
            else:
                print('send message to MQ failed.')
            if i == (retry_time - 1):
                print('send message to MQ failed after retries.')
    except Exception as e:
        print('ProducerSendSyncFailed:', e)
    finally:
        producer.shutdown()
        MUTEX.release()
        return


def send_orderly_with_sharding_key(count):
    producer = Producer(gid, True)
    producer.set_name_server_address(name_srv)
    producer.start()
    for n in range(count):
        msg = create_message()
        ret = producer.send_orderly_with_sharding_key(msg, 'orderId')
        print ('send message status: ' + str(ret.status) + ' msgId: ' + ret.msg_id)
    print ('send sync order message done')
    producer.shutdown()


def check_callback(msg):
    print ('check: ' + msg.body.decode('utf-8'))
    return TransactionStatus.COMMIT


def local_execute(msg, user_args):
    print ('local:   ' + msg.body.decode('utf-8'))
    return TransactionStatus.UNKNOWN


def send_transaction_message(count):
    producer = TransactionMQProducer(gid, check_callback)
    producer.set_name_server_address(name_srv)
    producer.start()
    for n in range(count):
        msg = create_message()
        ret = producer.send_message_in_transaction(msg, local_execute, None)
        print ('send message status: ' + str(ret.status) + ' msgId: ' + ret.msg_id)
    print ('send transaction message done')

    while True:
        time.sleep(3600)


if __name__ == '__main__':
    # send_message_sync(10)
    # send_message_multi_threaded(10)
    # thread_01 = Thread(target=send_message_multi_threaded(10))
    thread_02 = Thread(target=send_message_multi_threaded1(10)) 
    # thread_01.start()
    thread_02.start()


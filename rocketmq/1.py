from rocketmq.client import Producer, Message
import time

producer = Producer('producer_group')
producer.set_namesrv_addr('localhost:9876')
producer.start()

while True:
    message = Message('test_topic')
    message.set_body('1'.encode('utf-8'))
    producer.send_sync(message)
    time.sleep(1)

from rocketmq.client import PushConsumer

def print_message(msg):
    print(msg.body.decode('utf-8'))

consumer = PushConsumer('consumer_group')
consumer.set_namesrv_addr('localhost:9876')
consumer.subscribe('test_topic', '*', print_message)
consumer.start()

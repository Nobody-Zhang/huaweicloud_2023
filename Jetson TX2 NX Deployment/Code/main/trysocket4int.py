# coding=UTF-8
import socket
import threading 
import numpy as np 
import asyncio 
import websockets 
import logging 
import struct
from asyncio import run_coroutine_threadsafe  

logging.basicConfig(level=logging.INFO)  

flag = 0  # 控制传输
# front_end_socket = None  # 存储前端WebSocket连接
front_end_int_socket = None  # 存储前端WebSocket整数连接
record_socket = None  # 用于存储与4333端口的连接
event_loop = None  # asyncio事件循环对象



# 用于接收来自客户端的图像数据的线程函数
def receiver(conn, loop):
    global  front_end_int_socket, flag
    while True:
        int_data = conn.recv(4)  # 接收整数数据，4字节
        if not int_data:
            break
        int_value = struct.unpack('!I', int_data)[0]  # 使用big-endian解包整数值
        
        print('receive')
        print(int_value)
        if front_end_int_socket is not None:
            print('send')
            future = run_coroutine_threadsafe(send_int_to_frontend(int_value), loop)  # 在事件循环中运行协程
            future.result()
        else:
            print("尚未连接")

# 异步函数，用于将整数值发送给前端
async def send_int_to_frontend(int_value):
    await front_end_int_socket.send(str(int_value))  # 发送整数值
    print(f"Integer {int_value} sent to front end.")


# 用于处理整数值WebSocket通信的异步函数
async def int_stream(websocket, path):
    global front_end_int_socket
    front_end_int_socket = websocket
    while True:
        await asyncio.sleep(1)  # 保持连接活跃，每隔1秒发送心跳信号

if __name__ == '__main__':
    s = socket.socket()  # 创建一个新的socket对象
    host = 'localhost'
    port = 8764
    s.bind((host, port))  # 绑定主机和端口
    s.listen(5)  # 开始监听连接，最多允许5个连接
    conn, addr = s.accept()  # 接受客户端连接请求

    event_loop = asyncio.get_event_loop()  # 获取异步事件循环对象

    # 启动接收器线程
    receiver_thread = threading.Thread(target=receiver, args=(conn, event_loop))
    receiver_thread.start()

    logging.info("Starting WebSocket server")  # 记录日志，WebSocket服务器启动
    start_int_server = websockets.serve(int_stream, "0.0.0.0", 7980)  # 在0.0.0.0的7980端口上启动整数值WebSocket服务器
    asyncio.get_event_loop().run_until_complete(start_int_server)  # 运行整数值WebSocket服务器
    asyncio.get_event_loop().run_forever()  # 运行事件循环，保持程序运行状态


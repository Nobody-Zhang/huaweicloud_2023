import socket
import threading
import asyncio
import websockets
import logging
import struct
from asyncio import run_coroutine_threadsafe

logging.basicConfig(level=logging.INFO)

# 初始化两个与前端通信的Socket变量
front_end_int_socket_now = None   #8764
front_end_int_socket_previous = None   #8763

# 初始化事件循环
event_loop = None

# 用于接收来自客户端的图像数据的线程函数
def receiver(conn, loop, socket_id):
    global front_end_int_socket_now, front_end_int_socket_previous
    while True:
        int_data = conn.recv(4)
        if not int_data:
            break
        int_value = struct.unpack('!I', int_data)[0]

        # 根据Socket ID 将整数值发送给对应的前端
        if socket_id == 8764:
            if front_end_int_socket_now is not None:
                future = run_coroutine_threadsafe(send_int_to_frontend(int_value, front_end_int_socket_now), loop)
                future.result()
            else:
                print("尚未连接到8764端口的前端")
        elif socket_id == 8763:
            if front_end_int_socket_previous is not None:
                future = run_coroutine_threadsafe(send_int_to_frontend(int_value, front_end_int_socket_now), loop)
                future.result()
                print(int_value)
            else:
                print("尚未连接到8763端口的前端")

# 异步函数，用于将整数值发送给前端
async def send_int_to_frontend(int_value, websocket):
    await websocket.send(str(int_value))
    print(f"整数 {int_value} 已发送给前端.")

# 用于处理整数值WebSocket通信的异步函数
async def int_stream(websocket, path, socket_id):
    if socket_id == 7980:
        global front_end_int_socket_now
        front_end_int_socket_now = websocket
    elif socket_id == 7981:
        global front_end_int_socket_previous
        front_end_int_socket_previous = websocket

    while True:
        await asyncio.sleep(1)

# 用于初始化并接受socket连接的函数
def init_socket(port, loop):
    s = socket.socket()
    s.bind(('localhost', port))
    s.listen(5)
    conn, addr = s.accept()
    receiver_thread = threading.Thread(target=receiver, args=(conn, loop, port))
    receiver_thread.start()

if __name__ == '__main__':
    event_loop = asyncio.get_event_loop()

    # 在两个不同的线程中分别启动Socket 8764和8763的接收线程
    threading.Thread(target=init_socket, args=(8764, event_loop)).start()
    threading.Thread(target=init_socket, args=(8763, event_loop)).start()

    logging.info("开始WebSocket服务器")

    # 创建WebSocket服务器并启动
    start_int_server_7980 = websockets.serve(lambda ws, path: int_stream(ws, path, 7980), "0.0.0.0", 7980)   #now
    start_int_server_7981 = websockets.serve(lambda ws, path: int_stream(ws, path, 7981), "0.0.0.0", 7981)   #previous

    asyncio.get_event_loop().run_until_complete(start_int_server_7980)
    asyncio.get_event_loop().run_until_complete(start_int_server_7981)
    asyncio.get_event_loop().run_forever()

import socket
import threading
import numpy as np
import cv2
import asyncio
import websockets
import logging
import struct
from asyncio import run_coroutine_threadsafe

logging.basicConfig(level=logging.INFO)

flag = 0  # 是否传给前端
front_end_socket = None  # 链接
event_loop = None  # loop

# Thread for receiving images from the client and forwarding them to the frontend if flag is 1
def receiver(conn, loop):
    global front_end_socket
    global flag
    while True:
        size_data = conn.recv(4)
        if not size_data:
            break
        size = struct.unpack('!I', size_data)[0]

        img_data = b''
        while len(img_data) < size:
            part = conn.recv(size - len(img_data))
            if not part:
                break
            img_data += part

        img_array = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            print("Failed to decode image")
            continue

        if flag == 1 and front_end_socket is not None:
            future = run_coroutine_threadsafe(send_to_frontend(img), loop)
            future.result()

# 发送给前端
async def send_to_frontend(img):
    resized_img = cv2.resize(img, (480, 320))
    _, buffer = cv2.imencode('.jpg', resized_img)
    await front_end_socket.send(buffer.tobytes())
    print("Image sent to front end.")

# 接受前端控制
async def video_stream(websocket, path):
    global front_end_socket
    global flag
    front_end_socket = websocket
    while True:
        command = await websocket.recv()
        if command == 'start':
            flag = 1
        elif command == 'stop':
            flag = 0

if __name__ == '__main__':
    # 与deepstream建立连接
    s = socket.socket()
    host = 'localhost'
    port = 8765
    s.bind((host, port))
    s.listen(5)
    conn, addr = s.accept()

    event_loop = asyncio.get_event_loop()

    # 开始接受
    receiver_thread = threading.Thread(target=receiver, args=(conn, event_loop))
    receiver_thread.start()

    # 传输
    logging.info("Starting WebSocket server")
    start_server = websockets.serve(video_stream, "0.0.0.0", 7979)

    event_loop.run_until_complete(start_server)
    event_loop.run_forever()

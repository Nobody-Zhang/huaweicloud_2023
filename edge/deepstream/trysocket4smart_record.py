# coding=UTF-8
import socket
import threading
import numpy as np
import cv2
import asyncio
import websockets
import logging
import struct
import time  
from asyncio import run_coroutine_threadsafe

logging.basicConfig(level=logging.INFO)

flag = 0  # 是否传给前端
front_end_socket = None  # 链接
record_socket = None  # Placeholder for the socket object
event_loop = None  # loop

# Function to establish connection with localhost:4333
def establish_record_connection():
    global record_socket
    while True:
        try:
            record_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            record_socket.connect(('localhost', 4333))
            print("Connected to localhost:4333")
            break
        except ConnectionRefusedError:
            print("Connection refused. Retrying in 1 second.")
            time.sleep(1)

# Thread for receiving images from the client and forwarding them to the frontend if flag is 1
def receiver(conn,loop):
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

        print('receive')
        if flag == 1 and front_end_socket is not None:
            future = run_coroutine_threadsafe(send_to_frontend(img), loop)
            future.result()
        elif front_end_socket is None:
            print("尚未连接")
            

# Function to send the image to the frontend
async def send_to_frontend(img):
    resized_img = cv2.resize(img, (480, 320))
    _, buffer = cv2.imencode('.jpg', resized_img)
    await front_end_socket.send(buffer.tobytes())
    print("Image sent to front end.")

# Async function to handle WebSocket communication for setting flag
async def video_stream(websocket, path):
    global front_end_socket
    global flag
    front_end_socket = websocket
    while True:
        command = await websocket.recv()
        if command == 'start':
            flag = 1
            print('\n\n\n\n\n\n')
            print('\n\n\n\n\n\n')
        elif command == 'stop':
            flag = 0
            print('\n\n\n\n\n\n')
            print('stop')
            print('\n\n\n\n\n\n')
        elif command == 'record':  # New command added
            print('\n\n\n\n\n\n')
            print(2)
            print('\n\n\n\n\n\n\n')
            if record_socket is not None:
                record_socket.sendall(b'record')  # Send 'record' to the localhost:4333

if __name__ == '__main__':
    # Setup socket for receiving images
    s = socket.socket()
    host = 'localhost'
    port = 8765
    s.bind((host, port))
    s.listen(5)
    conn, addr = s.accept()

    event_loop = asyncio.get_event_loop()
    
    # 开始接受
    receiver_thread = threading.Thread(target=receiver, args=(conn,event_loop))
    receiver_thread.start()

    # 建立record连接
    record_socket_thread = threading.Thread(target=establish_record_connection)
    record_socket_thread.start()

    # 传输
    logging.info("Starting WebSocket server")
    start_server = websockets.serve(video_stream, "0.0.0.0", 7979)

    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()

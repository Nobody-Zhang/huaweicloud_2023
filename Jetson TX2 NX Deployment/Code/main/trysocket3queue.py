import socket
import threading
import collections
import asyncio
import websockets
import logging
import time
import struct
import numpy as np
import cv2

logging.basicConfig(level=logging.INFO)


class ThreadSafeQueue:
    def __init__(self):
        self.queue = collections.deque(maxlen=8)
        self.lock = threading.Lock()

    def put(self, item):
        with self.lock:
            if len(self.queue) == 8:  # 队列满了
                for _ in range(3):  # 从队尾出队 3 个元素
                    self.queue.pop()
            self.queue.append(item)  # 入队

    def get(self):
        with self.lock:
            if not self.queue:
                return None
            return self.queue.popleft()  # 从队首出队

    def empty(self):
        with self.lock:
            return not bool(self.queue)

    def qsize(self):
        with self.lock:
            return len(self.queue)

img_queue = ThreadSafeQueue()

def receiver(conn):
    while True:
        # First receive the size of the image
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
        else:
            img_queue.put(img)

        print(f"Received and added to queue. Current queue size: {img_queue.qsize()}")

async def video_stream(websocket, path):
    try:
        start_time = time.time()
        flag = 0
        k = 0
        fps = 10
        frame_duration = 1 / fps
        while True:
            if flag == 0:
                command = await websocket.recv()

            if command == 'start' or flag == 1:
                expected_time = start_time + (k * frame_duration)
                k += 1
                delay = expected_time - time.time()

                frame = img_queue.get()
                if frame is None:
                    logging.error("The queue is empty, skipping this frame.")
                    #await asyncio.sleep(0.1)
                    continue

                if delay < 0:
                    flag = 1
                    continue
                else:
                    await asyncio.sleep(delay)
                    flag = 0

                resized_frame = cv2.resize(frame, (480, 320))
                _, buffer = cv2.imencode('.jpg', resized_frame)
                await websocket.send(buffer.tobytes())

            elif command == 'stop':
                break

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    s = socket.socket()
    host = 'localhost'
    port = 8765
    s.bind((host, port))

    s.listen(5)
    conn, addr = s.accept()

    receiver_thread = threading.Thread(target=receiver, args=(conn,))
    receiver_thread.start()

    logging.info("Starting WebSocket server")
    start_server = websockets.serve(video_stream, "0.0.0.0", 7979)

    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()

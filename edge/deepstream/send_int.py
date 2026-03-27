import socket
import threading
import asyncio
import websockets
import logging
import struct
import time

logging.basicConfig(level=logging.INFO)

front_end_int_socket = None
conn = None
receiver_thread = None
event_loop = None
stop_receiver_flag = threading.Event()  # Added a flag to signal receiver to stop


def establish_warning_connection():
    global warning_socket
    while True:
        try:
            warning_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            warning_socket.connect(('localhost', 4332))
            print("Connected to localhost:4332")
            break
        except ConnectionRefusedError:
            print("Connection refused. Retrying in 1 second.")
            time.sleep(1)
            
            
def receiver(loop):
    global front_end_int_socket, conn, stop_receiver_flag
    try:
        while not stop_receiver_flag.is_set():  # Check the flag
            int_data = conn.recv(4)
            if not int_data:
                break
            int_value = struct.unpack('!I', int_data)[0]
            print('Received:', int_value)
            
            if warning_socket is not None:
                temp = 0
                if int_value >= 10:
                    temp = int_value%10
                int_bytes = temp.to_bytes(4, byteorder='big')
                warning_socket.send(int_bytes)

            if front_end_int_socket:
                future = asyncio.run_coroutine_threadsafe(send_int_to_frontend(int_value), loop)
                future.result()
    except Exception as e:
        print(f"An exception occurred in receiver: {e}")

async def send_int_to_frontend(int_value):
    await front_end_int_socket.send(str(int_value))
    print(f"Integer {int_value} sent to front end.")

async def int_stream(websocket, path):
    global front_end_int_socket, receiver_thread, event_loop, stop_receiver_flag
    front_end_int_socket = websocket

    # Signal old receiver thread to stop
    if receiver_thread and receiver_thread.is_alive():
        stop_receiver_flag.set()
        receiver_thread.join()

    # Reset flag and start new receiver thread
    stop_receiver_flag.clear()
    receiver_thread = threading.Thread(target=receiver, args=(event_loop,))
    receiver_thread.start()

    try:
        while True:
            await asyncio.sleep(1)
    except websockets.ConnectionClosed:
        print("WebSocket connection closed.")
        front_end_int_socket = None

async def start_websocket_server():
    await websockets.serve(int_stream, "0.0.0.0", 7980)
    print("WebSocket Server Started.")
    while True:
        await asyncio.sleep(1)

if __name__ == '__main__':
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
    host = 'localhost'
    port = 8764
    s.bind((host, port))
    s.listen(5)

    conn, addr = s.accept()
    print("Connected by", addr)
    
    record_warning_thread = threading.Thread(target=establish_warning_connection)
    record_warning_thread.start()

    event_loop = asyncio.get_event_loop()

    logging.info("Starting WebSocket server")
    event_loop.run_until_complete(start_websocket_server())
    event_loop.run_forever()

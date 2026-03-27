""" 
this is the very file that connect to the server and receive the warning type and finnaly give an answer to the user
"""
import socket
import threading
from recognize_generate4 import VoiceInteraction,establish_record_connection
import time 
# 创建VoiceInteraction对象
vi = VoiceInteraction()
warning_type = 0
last_type = -1

def handle_warning():
    global last_type
    while True:
        time.sleep(0.5)
        cur_type = warning_type
        if cur_type == 0:
            last_type = 0
        if cur_type != 0 and last_type != cur_type:
            vi.alert(cur_type)
            last_type = cur_type

s = socket.socket()
s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
host = 'localhost'
port = 4332
s.bind((host, port))
s.listen(5)
print("Waiting for connection..w.")
conn, addr = s.accept()
print("Connected by", addr)

def receive():
    global warning_type
    while True:
        wtype = conn.recv(4)
        warning_type = int.from_bytes(wtype, byteorder='big')
        print(warning_type)

def communicate():
    print('start communicate thread')
    vi.communicate()

if __name__ == "__main__":
    record_socket_thread = threading.Thread(target=establish_record_connection)
    record_socket_thread.start()
    microphone_thread = threading.Thread(target=communicate)
    receive_thread = threading.Thread(target=receive)
    handle_thread = threading.Thread(target=handle_warning)
    receive_thread.start()
    handle_thread.start()
    microphone_thread.start()




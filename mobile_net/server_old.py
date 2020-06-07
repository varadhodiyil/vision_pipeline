import os
import socket
import sys

import cv2
import numpy as np
from io import BytesIO
# from predict import PredictCars
#
# pred = PredictCars()

if not os.path.exists("tmp"):
    os.makedirs("tmp")

HOST = "localhost"
PORT = 5555

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(5)

print("Listening ...")

curr = 0
while True:
    try:
        conn, addr = s.accept()
        print("[+] Client connected: ", addr)
        tmp_path = "tmp/{0}.jpg".format(curr)
        f = open(tmp_path, "wb")
        while True:
            # get file bytes
            length = None
            ultimate_buffer = ""
            while True:
                data = conn.recv(1024)
                ultimate_buffer += data.decode()
                if len(ultimate_buffer) == length:
                    break
                while True:
                    if length is None:
                        if ':' not in ultimate_buffer:
                            break
                        # remove the length bytes from the front of ultimate_buffer
                        # leave any remaining bytes in the ultimate_buffer!
                        length_str, ignored, ultimate_buffer = ultimate_buffer.partition(':')
                        print(length_str)
                        length = int(length_str)
                    if len(ultimate_buffer) < length:
                        break
                    # split off the full message from the remaining bytes
                    # leave any remaining bytes in the ultimate_buffer!
                   
                    ultimate_buffer = ultimate_buffer[length:]
                    length = None
                    break
            final_image = np.load(BytesIO(ultimate_buffer))['frame']
            
            print(final_image.shape)
        
        # data = np.frombuffer(data)
        
        img = cv2.imread(tmp_path)
        print(img.shape)
        # print(type(data))
        img = cv2.resize(img, (224, 224))
        arr = [np.array(img)]
        arr = np.array(arr, dtype=float)

        # pred.predict(data)
        f.close()
        print("[+] Download complete!")
        curr = curr + 1
    except Exception as e:
        raise(e)
        print(e)
        # close connection
conn.close()
print("[-] Client disconnected")
sys.exit(0)

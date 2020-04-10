import cv2
import sys
from time import time as timer
# from predict import PredictCars , CLASSES
import numpy as np
import time
import csv
import socket
import sys
from io import BytesIO
import pickle


HOST = "localhost"
PORT = 5555



print("[+] Connected with Server")


def read_video(video_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    print("FPS" , fps)
    fps /= 1000
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            img = cv2.resize(frame , (224,224))
            # print(type(img))
            # f = BytesIO()
            # np.savez_compressed(f, frame=img)
            # f.seek(0)
            # out = f.read()
            # val = "{0}".format(len(f.getvalue()))  # prepend length of array
            # out = "{0}:{1}".format(val, out)
            s = socket.socket(socket.AF_INET,   socket.SOCK_STREAM)
            s.connect((HOST, PORT))
            serialized_data = pickle.dumps(img, protocol=2)
            s.sendall(serialized_data)
            s.close()
            print("sent To clf")

            s = socket.socket(socket.AF_INET,   socket.SOCK_STREAM)
            s.connect((HOST, 6666))
            serialized_data = pickle.dumps(img, protocol=2)
            s.sendall(serialized_data)
            s.close()
            print("sent to obj")

        else:
            break
        
    s.close()


read_video("m.mp4")
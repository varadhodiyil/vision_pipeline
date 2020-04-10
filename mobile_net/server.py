from __future__ import print_function

import socket
import sys
import threading

import numpy as np
from PIL import Image

from client_send_message import MessageSender
from predict import PredictCars

try:
    import cPickle as pickle
except ImportError:
    import pickle



pred = PredictCars()
s = socket.socket()
print("Socket Conn Started")
s.bind((b'', 5555))
s.listen(1)

sender = MessageSender()
idx = 0 
while True:
    c, a = s.accept()
    data = b''
    while True:
        block = c.recv(4096)
        if not block:
            break
        data += block
    c.close()
    if sys.version_info.major < 3:
        unserialized_input = pickle.loads(data)
    else:
        unserialized_input = pickle.loads(data, encoding='bytes')
    if unserialized_input is not None:
        # img = Image.fromarray(unserialized_input)
        img = unserialized_input.tolist()
        images = list()
        images.append(img)
        images = np.array(images, dtype=float)
        class_ =pred.predict(images)
        print(class_)
        sender.send_message("{0},car_type,{1}".format(idx,class_))
        idx = idx + 1
    # print(img.size)
    # img.show()

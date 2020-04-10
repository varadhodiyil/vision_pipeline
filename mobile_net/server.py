from __future__ import print_function

import socket
import sys
import threading

import numpy as np
from PIL import Image
from datetime import datetime

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
'''
Socket server to receive the cropped car image and send it to type classifier
'''
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
    # c.close()
    if sys.version_info.major < 3:
        unserialized_input = pickle.loads(data)
    else:
        unserialized_input = pickle.loads(data, encoding='bytes')
    if unserialized_input is not None:
        # img = Image.fromarray(unserialized_input)
        rec = datetime.now()
        img = unserialized_input.tolist()
        images = list()
        images.append(img)
        images = np.array(images, dtype=float)
        class_ =pred.predict(images)
        # print(class_)
        _proc = "{:f}".format(float((datetime.now() - rec).total_seconds()))
        resp_json = dict()
        resp_json['car_type'] = class_
        resp_json['car_type_time'] = _proc
        print("car_type",resp_json)
        # c.send_message("{0},car_type,{1},car_type_time,{2}".format(idx,class_,_proc))
        c.sendall(pickle.dumps(resp_json,protocol=2))
        c.close()
        idx = idx + 1
    # print(img.size)
    # img.show()

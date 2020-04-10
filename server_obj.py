from __future__ import print_function
import sys
import socket
import numpy as np
from PIL import Image
try:
    import cPickle as pickle
except ImportError:
    import pickle

import threading

from client_send_message import MessageSender

sender = MessageSender()


from yolo import YOLO
pred = YOLO()



s = socket.socket()
print("Socket Conn Started")
s.bind((b'',6666))
s.listen(1)
idx = 0
while True:
    c,a = s.accept()
    data = b''
    while True:
        block = c.recv(4096)
        if not block: break
        data += block
    c.close()
    if sys.version_info.major < 3:
        unserialized_input = pickle.loads(data)
    else:
        unserialized_input = pickle.loads(data,encoding='bytes')
    if unserialized_input is not None:
        img = Image.fromarray(unserialized_input)
        # img = unserialized_input
        # images = list()
        # images.append(img)
        # images = np.array(images,dtype=float)
        class_ = len(pred.detect_image(img))
        sender.send_message("{0},num_cars,{1}".format(idx,class_))
        idx = idx + 1
    # print(img.size)
    # img.show()
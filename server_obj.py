from __future__ import print_function
import sys
import socket
import numpy as np
from PIL import Image
try:
    import cPickle as pickle
except ImportError:
    import pickle
from datetime import datetime
import threading

from client_send_message import MessageSender
from color_classifier import detect_color


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
        rec = datetime.now()
        # img = unserialized_input
        # images = list()
        # images.append(img)
        # images = np.array(images,dtype=float)
        resp = pred.detect_image(img)
        class_ = len(resp)
        _proc = "{:f}".format(float((datetime.now() -rec).total_seconds()))
        sender.send_message("{0},num_cars,{1},num_cars_time,{2}".format(idx,class_,_proc))
        dete_colours = list()
        rec = datetime.now()
        for _i,_r in enumerate(resp):
                #print(_r)
                _img = img.crop(_r)
                _img = _img.resize((224,224))
                _img = np.array(_img)
                car_color = detect_color(_img)
                print(car_color)
                dete_colours.append(car_color)
        d_clr = ";".join(dete_colours)
        _proc = "{:f}".format(float((datetime.now() - rec).total_seconds()))
        sender.send_message("{0},colours,{1},colours_time,{2}".format(idx,d_clr,_proc))
        idx = idx + 1
    # print(img.size)
    # img.show()
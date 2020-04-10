import sys
import socket
import csv
from datetime import datetime
import pickle


class MessagerS():
    def __init(self , callback=None):
        s = socket.socket()
        print("Socket Conn Started")
        s.bind((b'',7777))
        s.listen(5)


        resp = list()
        # resp = np.zeros((1495)).tolist()
        for i in range(1495):
            resp.append({})
        try :
            while True:
                c,a = s.accept()
                
                message = ''
                while True:
                    data = c.recv(4096)
                    if not data: break
                    message += data.decode()
                _resp = json.
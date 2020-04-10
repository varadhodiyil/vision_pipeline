import socket
import numpy

try:
    import cPickle as pickle
except ImportError:
    import pickle

sock = socket.socket()
data= numpy.ones((1, 60))
sock.connect(('localhost',5555))
serialized_data = pickle.dumps(data, protocol=2)
sock.sendall(serialized_data)
sock.close()
sock = socket.socket()
data= numpy.zeros((1, 60))
sock.connect(('localhost',8000))
serialized_data = pickle.dumps(data, protocol=2)
sock.sendall(serialized_data)
sock.close()


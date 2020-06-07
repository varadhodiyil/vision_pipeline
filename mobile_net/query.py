import threading

class ReadInp(threading.Thread):
    def __init__(self):
        self.q = 1

    def run(self):
        while True:
            self.q = input()
            
class WriteIp(ReadInp):
    def run(self):
        while True:
            print(super(ReadInp).q)

t = ReadInp()
t.start()
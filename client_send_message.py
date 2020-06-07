import socket

class MessageSender():
    def __init__(self, host="localhost",port=7777):
        self.host = host
        self.port = port

    def send_message(self , message):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.host, self.port))
        sock.send(message.encode())
        sock.close()



if __name__ == "__main__":
    m = MessageSender()
    m.send_message("1,obj1,Hello")
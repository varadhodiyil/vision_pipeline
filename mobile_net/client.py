import socket
import sys


HOST = "localhost"
PORT = 5555

s = socket.socket(socket.AF_INET,   socket.SOCK_STREAM)
s.connect((HOST, PORT))
print("[+] Connected with Server")

# get file name to send
f_send = "_t/11.jpeg"
# open file
with open(f_send, "rb") as f:
    # send file
    print("[+] Sending file...")
    data = f.read()
    s.sendall(data)

    # close connection
    s.close()
    print("[-] Disconnected")
    sys.exit(0)
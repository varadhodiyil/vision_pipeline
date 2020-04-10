import sys
import socket
import csv


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
        _resp = message.split(",")
        print(message , len(_resp))
        if len(_resp) >= 2:
            idx = int(_resp[0])
            if idx not in resp:
                resp.insert(idx,{})
            resp[idx][_resp[1]] = _resp[2]
            resp[idx]['frame'] = idx
except KeyboardInterrupt:
    print("Exiting")
resp = list(filter(None,resp))
if len(resp) > 0:
    header = list(resp[0].keys())
    with open("resp.csv" , "w") as w:
        writer = csv.DictWriter(w,fieldnames=header)
        writer.writeheader()
        writer.writerows(resp)
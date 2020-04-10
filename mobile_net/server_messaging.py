import sys
import socket
import csv
from datetime import datetime

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
        if len(_resp) >= 4:
            idx = int(_resp[0])
            resp[idx][_resp[1]] = _resp[2]
            resp[idx]['frame'] = idx
            # _rec = datetime.fromtimestamp(int(_resp[4]))
            # print((datetime.now() - _rec).total_seconds())
            resp[idx][_resp[3]] = str(_resp[4])
        print(resp[idx] , _resp[1])
        if len(list(filter(None,resp))) >= 1495:
            break
except KeyboardInterrupt:
    print("Exiting")
resp = list(filter(None,resp))
if len(sys.argv) >1 :
    out_file = sys.argv[1]
else:
    out_file = "resp.csv"
if len(resp) > 0:
    header = list(resp[0].keys())
    print(header)
    with open(out_file , "w") as w:
        writer = csv.DictWriter(w,fieldnames=header)
        writer.writeheader()
        for row in resp:
            if 'num_cars' not in row:
                row['num_cars'] = 0
            if 'car_type'  in row:
                # row['car_type'] = ""
                if row['num_cars'] == '0' or row['num_cars'] == 0:
                    row['car_type'] = ""
            writer.writerow(row)
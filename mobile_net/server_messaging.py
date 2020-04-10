import sys
import socket
import csv
from datetime import datetime
import pickle

s = socket.socket()
print("Socket Conn Started")
s.bind((b'',7777))
s.listen(5)








resp = list()
# resp = np.zeros((1495)).tolist()
# for i in range(1495):
#     resp.append({})
'''
Socket server to recieve response data and writes it into csv
'''
try :
    while True:
        c,a = s.accept()
        
        message = list()
        while True:
            data = c.recv(4096)
            if not data: break
            message.append(data)
        c.close()
        final = dict()
        message = b"".join(message)
        if sys.version_info.major < 3:
            final = pickle.loads(message)
        else:
            final = pickle.loads(message,encoding='bytes')
        resp.append(final)
        # _resp = message.split(",")
        
        # print(message , len(_resp))
        # if len(_resp) >= 4:
        #     idx = int(_resp[0])
        #     resp[idx][_resp[1]] = _resp[2]
        #     if _resp[1] == 'car_type':
        #         print(idx , message)
        #     resp[idx]['frame'] = idx
        #     # _rec = datetime.fromtimestamp(int(_resp[4]))
        #     # print((datetime.now() - _rec).total_seconds())
        #     resp[idx][_resp[3]] = str(_resp[4])
        # print(resp[idx] , _resp[1])
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
    header = ["num_cars","frame","num_cars_time","colours","colours_time","car_type","car_type_time"]
    print(header)
    with open(out_file , "w") as w:
        writer = csv.DictWriter(w,fieldnames=header)
        writer.writeheader()
        for row in resp:
            for h in header:
                if h not in row:
                    row[h] = None
            if 'num_cars' not in row:
                row['num_cars'] = 0
            if 'car_type'  in row:
                # row['car_type'] = ""
                if row['num_cars'] == '0' or row['num_cars'] == 0:
                    row['car_type'] = None
            writer.writerow(row)
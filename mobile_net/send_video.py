import cv2
import sys
from time import time as timer
# from predict import PredictCars , CLASSES
import numpy as np
import time
import csv
import socket
import sys
from io import BytesIO
import pickle
from datetime import datetime
import os

from client_send_message import MessageSender
# from scipy.misc import toimage
from PIL import Image

start = datetime.now()
HOST = "localhost"
PORT = 5555
sender = MessageSender()



print("[+] Connected with Server")
q = "q3"

def read_video(video_path):
    '''
    Method to read video
    video_path  : path of the video to be read

    '''
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise IOError("Couldn't open webcam or video")
    fps = video.get(cv2.CAP_PROP_FPS)
    print("FPS" , fps)
    fps /= 1000
    idx = 0
    # video_FourCC    = int(video.get(cv2.CAP_PROP_FOURCC))
    video_FourCC = cv2.VideoWriter_fourcc(*"mp4v")
    video_fps       = video.get(cv2.CAP_PROP_FPS)   
    video_size      = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    file_name , ext = os.path.splitext(video_path)
    out_path = "{0}_out{1}".format(file_name,ext)     
    print(out_path)
    out_video_obj = cv2.VideoWriter(out_path, video_FourCC, 25, video_size)
    while True:
        ret, frame = video.read()
        final_message = ""
        if ret:
            _frame = np.array(frame)
            img = cv2.resize(frame , (224,224))
            # print(type(img))
            # f = BytesIO()
            # np.savez_compressed(f, frame=img)
            # f.seek(0)
            # out = f.read()
            # val = "{0}".format(len(f.getvalue()))  # prepend length of array
            # out = "{0}:{1}".format(val, out)

            #block to put frames into queue and read result from object detection
            if q in ["q1","q2","q3"]:
                s = socket.socket(socket.AF_INET,   socket.SOCK_STREAM)
                s.connect((HOST, 6666))
                serialized_data = pickle.dumps(img, protocol=2)
                s.sendall(serialized_data)
                # s.shutdown(socket.SHUT_WR)
                # s.sendall('EOF'.encode("ascii"))
                # print("sent to obj")
                s.shutdown(socket.SHUT_WR)
                # s1 = socket.socket(socket.AF_INET,   socket.SOCK_STREAM)
                # s1.connect((HOST, 6666))
                # # s1.listen(4)
                # print("waiting....!")
                data = b''
                # while True:
                block = s.recv(4096)
                    # print(block)
                    # if not block: 
                    #     break
                    # data += block
                    # # s.shutdown(socket.SHUT_WR)
                data = block
                s.close()
                hasCar = {}
                if sys.version_info.major < 3:
                    hasCar = pickle.loads(data)
                else:
                    hasCar = pickle.loads(data,encoding='bytes')
                final = dict()
                final.update(hasCar)
                car_type = {}
                final_message += " Num Cars {0} \n".format(final['num_cars'])
                if q == "q3":
                    final_message += " Colours {0} \n".format(final['colours'])
            #After object detection, If query is Q2/Q3, detected car will send to Classifiers
            if q in ["q2","q3"]:
                if 'has_car' in hasCar and hasCar['has_car']:
                    print("car Found... Sending to Clf")
                    
                    s1 = socket.socket(socket.AF_INET,   socket.SOCK_STREAM)
                    s1.connect((HOST, PORT))
                    serialized_data = pickle.dumps(img, protocol=2)
                    s1.sendall(serialized_data)
                    s1.shutdown(socket.SHUT_WR)
                    block = s1.recv(4096)
                    s1.close()
                    
                    if sys.version_info.major < 3:
                        car_type = pickle.loads(block)
                    else:
                        car_type = pickle.loads(block,encoding='bytes')
                    final_message += "Car Type {0}".format(car_type['car_type'])
                    # print("car_type",car_type)
                # s.close()
                
                
                    print("sent To clf")
                else:
                    # sender.send_message("{0},car_type,{1},car_type_time,{2}".format(idx,"",""))
                    car_type['car_type'] = ""
                    car_type['car_type_time'] = ""
                    final_message += "No car found"

            final['frame'] = idx
            
            final.update(car_type)
            # print(final)
            final.pop('has_car')
            sender.send_message(pickle.dumps(final,protocol=2))
            idx = idx + 1
            # img = np.array(img)
            y0, dy = 20, 15
            for i, line in enumerate(final_message.split('\n')):
                y = y0 + i * dy
                cv2.putText(_frame, text=line, org=(10, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(255, 0, 0), thickness=2)
            # cv2.imshow("Result",img)
            print("Writing")
            # print(type(img))
            # _img = scipy.misc.toimage(img)
            # _img = Image.fromarray(img)

            out_video_obj.write(_frame)
            # _img.show()
            # if idx == 100:
            #     break
            

        else:
            break
    # out_video_obj
    out_video_obj.release()
        
   

read_video("m.mp4")

end = datetime.now()
dff = (end - start).total_seconds()

print("Time Takes for 1495 frames", dff)
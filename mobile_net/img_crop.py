from yolo import YOLO
import color_classifier as cc
import cv2
import numpy as np
from PIL import Image
import os
if __name__ == "__main__":

    y = YOLO()
    car_color_list = []
    i = 0
    for r, _ , files in os.walk(r"../../dataset"):
        for f in files:
            # print()
            in_path = os.path.joinappend(r,f)
            out_path = in_path.replace("dataset","cropped")
            #print("file",in_path)
            if not os.path.exists(os.path.dirname(out_path)):
                os.makedirs(os.path.dirname(out_path))
            img = Image.open(in_path)
            resp = y.detect_image(img)
            for _i,_r in enumerate(resp):
                #print(_r)
                _img = img.crop(_r)
                _img = _img.resize((224,224))
                op , ext = os.path.splitext(out_path)
                op = op +"_{}".format(_i) + "." + ext 
                print(op)
                _img.save(op)
                _img = np.array(_img)
                car_color = cc.detect_color(_img)
                print(car_color)
                car_color_list.append(car_color)
            
            i = i + 1
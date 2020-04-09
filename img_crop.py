from yolo import YOLO
import cv2
import numpy as np
from PIL import Image
import os
if __name__ == "__main__":

    y = YOLO()
    i = 0
    for r, _ , files in os.walk("mobile_net/data"):
        for f in files:
            # print()
            in_path = os.path.join(r,f)
            out_path = in_path.replace("data","cropped")
            print("file",in_path)
            if not os.path.exists(os.path.dirname(out_path)):
                os.makedirs(os.path.dirname(out_path))
            img = Image.open(in_path)
            resp = y.detect_image(img)
            for _i,_r in enumerate(resp):
                _img = img.crop(_r)
                _img = _img.resize((224,224))
                op , ext = os.path.splitext(out_path)
                op = op +"_{}".format(_i) + "." + ext 
                print(op)
                _img.save(op)
            
            if i == 20:
                break
            i = i + 1
    
    # img = cv2.imread("test.jpg")
    # img = Image.open("test.jpg")
    # img = np.array(img)
    # print(img.shape)
    # resp = y.detect_image(img)
    # resp = [(218, 120 - 20,356, 181)]
    # for r in resp:
    #     # print(r)
    #     _img = img.crop(r)
    #     _img = _img.resize((224,224))
    #     # cv2.imshow("A" , _img)
    #     _img.show()


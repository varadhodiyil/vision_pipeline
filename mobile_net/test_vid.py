import cv2
import sys
from time import time as timer
from predict import PredictCars , CLASSES
import numpy as np
import time
import csv
# import pickle

# CLASSES = pickle.load( open("mapping","rb"))




pr = PredictCars()

def read_video(video_path):
    to_csv = list()
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    print("FPS" , fps)
    fps /= 1000
    framerate = timer()
    elapsed = int()
    # cv2.namedWindow('ca1', 0)
    out_p = list()
    i = 0
    while video.isOpened():

        start = timer()
        # print(start)
        ret, frame = video.read()
        if ret:
            img = cv2.resize(frame , (224,224))
            arr = [ np.array(img)]
            arr = np.array(arr , dtype=float)
            _pred = pr.predict(arr)
            out = "DEt:  %s" % _pred
            out_p.append(_pred)
            to_csv.append([str(CLASSES[_pred])])
            # cv2.imshow('ca1',frame)
            # cv2.imwrite('test.jpg',frame)
            # print("saved")
            # sys.exit(0)
            # cv2.putText(img, text=out, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #             fontScale=0.50, color=(255, 0, 0), thickness=2)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            # cv2.imshow('F',img)
            # time.sleep(0.5)
            # diff = timer() - start
            # while  diff < fps:
            #     diff = timer() - start

            elapsed += 1
            if elapsed % 5 == 0:
                sys.stdout.write('\r')
                sys.stdout.write('{0:3.3f} FPS'.format(elapsed / (timer() - framerate)))
                sys.stdout.flush()
            if i%100 ==0:
                print("Done Frame ", i)
            i = i + 1
        else:
            break
    print(set(out_p))
    video.release()
    # cv2.destroyAllWindows()
    with open("clf.csv","w") as w:
        writer = csv.writer(w)
        # print(to_csv)
        writer.writerows(to_csv)



read_video("video.mp4")
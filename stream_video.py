import cv2
import sys
from time import time as timer

def read_video(video_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    fps /= 1000
    framerate = timer()
    elapsed = int()
    cv2.namedWindow('ca1', 0)
    while(video.isOpened()):

        start = timer()
        # print(start)
        ret, frame = video.read()
        if ret:
            cv2.imshow('ca1',frame)
            cv2.imwrite('test.jpg',frame)
            print("saved")
            sys.exit(0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            diff = timer() - start
            while  diff < fps:
                diff = timer() - start

            elapsed += 1
            if elapsed % 5 == 0:
                sys.stdout.write('\r')
                sys.stdout.write('{0:3.3f} FPS'.format(elapsed / (timer() - framerate)))
                sys.stdout.flush()

    video.release()
    cv2.destroyAllWindows()


read_video("../video.mp4")
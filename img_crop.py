from yolo import YOLO
import cv2
import numpy as np
from PIL import Image

if __name__ == "__main__":

    y = YOLO()
    # img = cv2.imread("test.jpg")
    img = Image.open("test.jpg")
    # img = np.array(img)
    # print(img.shape)
    print("ret",y.detect_image(img))


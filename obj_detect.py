from PIL import Image
from yolo import YOLO


yolo = YOLO(model="model_data/tiny.h5",anchors="model_data/tiny_yolo_anchors.txt")


def detect_img(path):
    img = Image.open(path)

    yolo.detect_image(img)


detect_img('test.jpg')
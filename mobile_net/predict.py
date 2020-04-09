import tensorflow as tf
import numpy as np
import cv2
CLASS_NAMES = ["sedan", "hatchback"]
class PredictCars():
    def __init__(self):
        self.model = tf.keras.models.load_model("car")
        self.model.build((None,224,224,3))
        print(self.model.summary())


    def predict(self , images):
        # images = tf.keras.preprocessing.image.img_to_array(images)
        print(images.shape)
        ret = np.argmax(self.model.predict(images), axis=1)

        _ret = list()
        for r in ret:
            _ret.append(CLASS_NAMES[r])
        
        return " ,".join(_ret)




if __name__ == "__main__":
    p = PredictCars()
    # img = Image.open("../test.jpg")
    # img = cv2.imread("../test.jpg")
    img = cv2.resize(img, (320, 320))
    arr = [ np.array(img)]
    arr = np.array(arr , dtype=float)
    # print(arr.shape)
    print(p.predict(arr))

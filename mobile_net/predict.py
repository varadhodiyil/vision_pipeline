import tensorflow as tf
import numpy as np
import cv2
import os

# CLASS_NAMES = ["sedan", "hatchback"]
import pickle
CLASSES = pickle.load( open("mapping","rb"))
print(CLASSES)

class PredictCars():
    def __init__(self):
        self.model = tf.keras.models.load_model("car")
        self.model.build((None,224,224,3))
        print(self.model.summary())


    def predict(self , images):
        # images = tf.keras.preprocessing.image.img_to_array(images)
        ret = np.argmax(self.model.predict(images), axis=1)

        _ret = list()
        for r in ret:
            _ret.append(list(CLASSES.keys())[list(CLASSES.values()).index(r)])
        
        return " ,".join(_ret)




if __name__ == "__main__":
    p = PredictCars()
    for r , _ , files in os.walk("_t"):
        print(r)
        for f in files:
            # img = Image.open("../t/11.jpeg")
            path = os.path.join(r,f)
            print(path)
            img = cv2.imread(path)
            img = cv2.resize(img, (224, 224))
            arr = [ np.array(img)]
            arr = np.array(arr , dtype=float)
            # print(arr.shape)
            print(p.predict(arr))

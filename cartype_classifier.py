from keras.models import model_from_json
import numpy as np
import cv2
from keras.preprocessing import image

class CarTypeClassifier:
    def __init__(self):
        json_file = open('mobile_net/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = model_from_json(loaded_model_json)
        self.loaded_model.load_weights("mobile_net/model.h5",by_name=True) 
        self.loaded_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
       
    def load_image(self, img, show=False): 
        img = cv2.resize(img, (256, 256))
        img_tensor = image.img_to_array(img)                    # (height, width, channels)
        img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
        img_tensor /= 255.                                      # imshow expects values in the range [0, 1]
        return img_tensor

    def detect_cartype(self, img):
        img = self.load_image(img)                    
        pred = self.loaded_model.predict(img)
        return pred
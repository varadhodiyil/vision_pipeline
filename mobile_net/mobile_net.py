import tensorflow as tf
import numpy as np 
from glob import glob
import os

from PIL import Image

IMG_SIZE = 320
IMG_SHAPE = (IMG_SIZE , IMG_SIZE , 3)

def resize_img(image):
    print(image)
    img = Image.open(image)
    img = img.resize((256,256) , Image.ANTIALIAS)

    img.save(image)

def format_example(image, label):
  image = tf.cast(image, tf.float32)
  image = (image/127.5) - 1
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
  return image, label

# for r,_,files in os.walk("data"):
#     for f in files:
#         # print(r,f)
#         resize_img(os.path.join(r,f))


base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

base_model.summary()


import pathlib
data_dir = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                                         fname='flower_photos', untar=True)
data_dir = pathlib.Path(data_dir)
print(data_dir)


BATCH_SIZE = 10
IMG_HEIGHT = 256
IMG_WIDTH = 256
image_count = 1110
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

image_generator = tf.keras.preprocessing.image.ImageDataGenerator()

CLASS_NAMES = ["sedan", "hatchback"]
# data_dir = glob("**/*.jpg",recursive=True)
# print(data_dir)

data_dir = "data"

train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes = list(CLASS_NAMES))

test_data_gen = image_generator.flow_from_directory(directory=str("test"),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes = list(CLASS_NAMES))
# image_batch, label_batch = next(train_data_gen)
# print(image_batch , label_batch)

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')


image_batch, label_batch = next(train_data_gen)
feature_batch = base_model(image_batch)

base_model.trainable = False
print(base_model.summary())
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

prediction_layer = tf.keras.layers.Dense(2 , activation='softmax')
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  tf.keras.layers.Dense(1280,activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1280,activation='relu'),
  
  # tf.keras.layers.Dense(640,activation='relu'),
  prediction_layer
])


base_learning_rate = 0.0001
model.compile(
              #optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              # loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])



initial_epochs = 15
validation_steps=20

# loss0,accuracy0 = model.evaluate(validation_batches, steps = validation_steps)

history = model.fit(train_data_gen,
                    epochs=initial_epochs,
                    validation_data=test_data_gen)



acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

model.save('car')
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


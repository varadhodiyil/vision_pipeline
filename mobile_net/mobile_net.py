import tensorflow as tf
import numpy as np 
from glob import glob
import os

from PIL import Image
from datetime import datetime
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pickle

IMG_SIZE = 224
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


# import pathlib
# data_dir = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
#                                          fname='flower_photos', untar=True)
# data_dir = pathlib.Path(data_dir)
# print(data_dir)


BATCH_SIZE = 10
IMG_HEIGHT = 256
IMG_WIDTH = 256
image_count = 1110
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.2,preprocessing_function=preprocess_input)

CLASS_NAMES = ["sedan", "hatchback"]
# data_dir = glob("**/*.jpg",recursive=True)
# print(data_dir)

data_dir = "data"

train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                    #  classes = list(CLASS_NAMES),
                                                      subset="training",
                                                      class_mode='categorical')
label_map = (train_data_gen.class_indices)
print(label_map)
test_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                    #  classes = list(CLASS_NAMES),
                                                     class_mode='categorical',
                                                     subset="validation")

label_map_t = (test_data_gen.class_indices)
print(label_map_t)


pickle.dump(label_map,open("mapping","wb"))
# image_batch, label_batch = next(train_data_gen)
# print(image_batch , label_batch)

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)


logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# Create the base model from the pre-trained model MobileNet V2

base_model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=IMG_SHAPE)

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
# x = Dropout(rate = .2)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(1280, activation='relu',  kernel_initializer= tf.keras.initializers.glorot_uniform(42), bias_initializer='zeros')(x)
# x = Dropout(rate = .2)(x)
x = tf.keras.layers.BatchNormalization()(x)
predictions = tf.keras.layers.Dense(2, activation='softmax', kernel_initializer='random_uniform', bias_initializer='zeros')(x)

model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

base_learning_rate = 0.0001
optimizer = tf.keras.optimizers.Adam(lr=base_learning_rate)
# optimizer = RMSprop(lr=learning_rate)

loss = "categorical_crossentropy"
# loss = "kullback_leibler_divergence"

for layer in model.layers:
    layer.trainable = True
# for layer in model.layers[-2:]:
#     layer.trainable = True


model.summary()

model.compile(
              #optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              # loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])



initial_epochs = 10
validation_steps=20

# loss0,accuracy0 = model.evaluate(validation_batches, steps = validation_steps)

history = model.fit(train_data_gen,
                    epochs=initial_epochs,
                    validation_size=0.2,
                    #validation_data=test_data_gen,
                    callbacks=[tensorboard_callback])



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


import tensorflow as tf
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
from tensorflow import keras

mnist = tf.keras.datasets.mnist
(x_train,y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model =tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(40,40)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(18, activation='softmax'))

model.compile(optimizer='adam', loss ='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)
model.save('handwritten.model')

model= tf.keras.models.load_model('handwritten.model')
image_number = 1
while os.path.isfile("f/home/kshubh5/Downloads/Untitled design(1){image_number}.jpg"):
  try:
    img = cv.imread("/home/kshubh5/Downloads/Untitled design(1){image_number}.jpg")[:,:,0]
    img = np.invert(np.array([img]))
    prediction = model.predict('img')
    print('this is probably a {np.argmax(prediction)}')
    plt.ishow('img',[0], cmap =plt.cm.binary)
    plt.show()
  except:
    print('Error')

  finally:
      image_number=1






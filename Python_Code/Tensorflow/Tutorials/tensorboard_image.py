
from datetime import datetime
import io
import itertools
from packaging import version

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import cv2

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics

mnist = tf.keras.datasets.mnist
# Download the data. The data is already divided into train and test.
# The labels are integers representing classes.
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape the image for the summary API.
img = np.reshape(x_train[0], (-1,28,28,1))

# Sets up a timestamped log directory.
logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# Creates a file writer for the log directory.
file_writer = tf.summary.create_file_writer(logdir)

class LoG(keras.layers.Layer):
    def __init__(self):
        super(LoG, self).__init__()
        tf.config.run_functions_eagerly(True)

    def call(self, x):
        return self.filter(x)

    def filter(self, input):


        input = np.reshape(input, (128,196,1))
        print( input.shape)
        x = input
        cv2.imshow("test", x)
        # Apply Gaussian Blur to the Image
        blur = cv2.GaussianBlur(x, (3, 3), 1)

        # Apply Laplacian Filter to the Image
        laplacian = cv2.Laplacian(blur, cv2.CV_8UC1, ksize=9, delta=1)

        thresh_img = np.zeros(blur.shape)

        # finding the contours of the laplacian filter
        ret, thresh = cv2.threshold(laplacian, 0, 0, 0)
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        # draw the contours on the empty image
        cnt = cv2.drawContours(thresh_img, contours, -1, (0, 255, 0), 1)
        cv2.imshow("tst", laplacian)
        cv2.waitKey(0)
        cnt = tf.convert_to_tensor(laplacian, dtype=tf.float32)
        return laplacian

# Using the file writer, log the reshaped image.
#------------------------------------------------------------------------------
#Model Definition
#------------------------------------------------------------------------------
model = models.Sequential()
model.add(LoG())

def show_image(epoch, logs):
    with file_writer.as_default():
      tf.summary.image("Training data", img, step=epoch)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
per_epoch = keras.callbacks.LambdaCallback(on_epoch_end=show_image)
model.fit(x=x_train,
          y=y_train,
          epochs=5,
          validation_data=(x_test, y_test),
          callbacks=[per_epoch])



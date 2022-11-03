import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy
import cv2



(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


def log_layer(x):

    # normalize the colors in the image
    x_np = x.numpy()
    imgray = cv2.cvtColor(x_np, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to the Image
    blur = cv2.GaussianBlur(x_np, (3, 3), 1)

    # Apply Laplacian Filter to the Image
    laplacian = cv2.Laplacian(blur, cv2.CV_32F, ksize=9, delta=1)

    return laplacian

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(tf.keras.layers.Lambda(log_layer))
model.add(layers.Flatten())
model.add(layers.Dense(10))


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'], run_eagerly=True)

history = model.fit(train_images, train_labels, epochs=1, validation_data=(test_images, test_labels))


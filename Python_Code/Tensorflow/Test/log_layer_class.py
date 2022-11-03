"""
Creator: Josiah Burnam
Org:     FGCU Visual Psychophysics Research
Desc:    Lapalcian of Gaussian Layer for Tensorflow 2.x

TODO:
    @JosiahBurnham
    Visualize the effects of the log operator on the fashion MNIST dataset (possibly with tensorboard)

TODO:
    @JosiahBurnam
    convert tensor to numpy array without eager execution (may not be possible)
"""

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import keras.preprocessing as preprocess
import cv2
import matplotlib.pyplot as pyplot
import numpy as np


#------------------------------------------------------------------------------
#Custom LoG Layer
#------------------------------------------------------------------------------
class LoG(keras.layers.Layer):
    def __init__(self):
        super(LoG, self).__init__()
        tf.config.run_functions_eagerly(True)

    def call(self, x):
        return self.filter(x)

    def filter(self, input):

        x = input.numpy()
        # Apply Gaussian Blur to the Image
        blur = cv2.GaussianBlur(x, (3, 3), 1)

        # Apply Laplacian Filter to the Image
        laplacian = cv2.Laplacian(blur, cv2.CV_8UC1, ksize=9, delta=1)
        laplacian = tf.convert_to_tensor(laplacian, dtype=tf.float32)
        return laplacian

#------------------------------------------------------------------------------
#Dataset set up
#------------------------------------------------------------------------------
(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

#------------------------------------------------------------------------------
#Model Definition
#------------------------------------------------------------------------------
model = models.Sequential()
model.add(layers.Flatten())
model.add(LoG())

#------------------------------------------------------------------------------
#Model compilation and Training
#------------------------------------------------------------------------------

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'], run_eagerly=True)

history = model.fit(train_images, train_labels, epochs=1, validation_data=(test_images, test_labels))



#------------------------------------------------------------------------------
# Visualize the data after model
#------------------------------------------------------------------------------

"""
img = preprocess.image.load_img("C:\\Users\\jjburnham0705\\Desktop\\Image_Filtering\\DetectEdgesInImagesExample_01.png",
                                target_size=(150,150))
img_tensor = preprocess.image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor = img_tensor / 255.0

layer_outputs = [layer.output for layer in model.layers[:1]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(img_tensor)

# Getting Activations of first layer
first_layer_activation = activations[0]

# shape of first layer activation
print(first_layer_activation.shape)

# 6th channel of the image after first layer of convolution is applied
plt.matshow(first_layer_activation[0, :, :, 6], cmap='viridis')

# 15th channel of the image after first layer of convolution is applied
plt.matshow(first_layer_activation[0, :, :, 15], cmap='viridis')
"""




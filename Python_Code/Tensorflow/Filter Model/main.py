import tensorflow as tf
import tensorflow.keras.preprocessing
import numpy as np
import matplotlib.pyplot as plt

from filter_bank import FilterBank


og_img = tensorflow.keras.preprocessing.image.load_img("test_pic.jpg")
img = np.float32(og_img)  # convert uint8 into float32
img = tf.image.rgb_to_grayscale(img)
img = img / 255.0  # normalize the image
expanded_img = tf.expand_dims(img, axis=0)


tf.config.run_functions_eagerly(True)

# Set up filters
filters = [["first_stage_32.mat", "filter_mat", 26],
           ["first_stage_16.mat", "filter_mat", 26],
           ["first_stage_8.mat", "filter_mat", 26],
           ["first_stage_4.mat", "filter_mat", 26]]

filter_bank = FilterBank(filters, pool_ksize=15, pool_strides=15)

filter_img = filter_bank(expanded_img)

img = filter_img[:, :, :, 0]
img = tf.reshape(img, shape=(100, 100, 1))
plt.imshow(img)
plt.show()

print(filter_img.shape)



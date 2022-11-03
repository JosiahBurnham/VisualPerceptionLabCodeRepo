"""
file:   fixed_filter_layer.py
by:     Josiah Burnham
org:    FGCU
desc:   A fixed filter layer that can be convolved with an image
"""
import tensorflow as tf
from tensorflow import keras
import scipy.io

"""
This class inherits from the Keras subclassing API
more information can be found here:https://www.tensorflow.org/guide/keras/custom_layers_and_models
"""


class FilterLayer(keras.layers.Layer):
    """
    Creates a Fixed Filter Layer
    """

    def __init__(self, filter, strides=1):
        """
        standard initializer for the FixedLayer class

        :param strides:  the strides the convolutional filter uses in it's calculation
        """
        super(FilterLayer, self).__init__()

        self.strides = strides
        # adds a output channel dimension to the filter layer with a value of one (which means it will be in grey scale)
        self.filter = tf.expand_dims(filter, axis=3)

    def call(self, x):
        """
        The function that is run when the class class object is called

        :param x: the image(s) to be convolved with the fixed filter
        :return: the convolved images
        """
        x = self._convolution(x)
        x = self._max_pool(x)
        return x

    def _convolution(self, input_data):
        """
        the calculation to convolve the fixed filter and the image

        :param input_data: the image that is to be convolved with the filter
        :return: the rectified linear image that was convolved with the filter
        """
        input_data = tf.nn.conv2d(input_data, self.filter, strides=[1, self.strides, self.strides, 1],
                                  padding="SAME")
        return tf.nn.relu(input_data)

    @staticmethod
    def _max_pool( input_img):
       return  tf.nn.max_pool(input_img, ksize=[1,4,4,1], strides=[1,4,4,1], padding="SAME" )

#==========================================================
# File  :   VGG19.py
# Author:   J.Burnham
# Date  :   01/31/2022
# Purpose:  Implement the VGG16 Convolutional Neural Network into Keras
#==========================================================

from tensorflow import keras
from tensorflow.keras import layers

from Layers.imbedRGB import ImbedRGB


class VGG_19(): 
    """ A implementation of the VGG19 Model, where the caller can specify what block
        of the model is desired.
    """
    def __init__(self, wd, num_outputs, output=True):
        """property constructor for VGG_19 class

        Args:
            wd (array[numpy arrays]): weight array of numpy arrays
            num_outputs (int): number of perdiction classes
        """
        self.num_outputs = num_outputs
        self.wd = wd 
        self.output = output

        # Conv Layers
        #----------------------------------------
        self.conv1 = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation="relu", name="conv1", trainable=False)
        self.conv2 = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation="relu", name="conv2", trainable=False)
        self.conv3 = layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', activation="relu", name="conv3", trainable=False)
        self.conv4 = layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', activation="relu", name="conv4", trainable=False)
        self.conv5 = layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation="relu", name="conv5", trainable=False)
        self.conv6 = layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation="relu", name="conv6", trainable=False)
        self.conv7 = layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation="relu", name="conv7", trainable=False)
        self.conv8 = layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation="relu", name="conv8", trainable=False)
        self.conv9 = layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation="relu", name="conv9", trainable=False)
        self.conv10 = layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation="relu", name="conv10", trainable=False)
        self.conv11 = layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation="relu", name="conv11", trainable=False)
        self.conv12 = layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation="relu", name="conv12", trainable=False)
        self.conv13 = layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation="relu", name="conv13", trainable=False)
        self.conv14 = layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation="relu", name="conv14", trainable=False)
        self.conv15 = layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation="relu", name="conv15", trainable=False)
        self.conv16 = layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation="relu", name="conv16", trainable=False)

        # Fully Connected Layers
        #----------------------------------------
        self.hidden_dense1 = layers.Dense(4096, activation="relu", name="hidden1", trainable=False)
        self.hidden_dense2 = layers.Dense(4096, activation="relu", name="hidden2", trainable=False)
        self.output_layer = layers.Dense(units=num_outputs, activation="softmax", name="output_layer", trainable=True)

        # Intermediary Layers
        #----------------------------------------
        self.flatten = layers.Flatten()
        self.imbedRGB = ImbedRGB(227,227, name="ImbedRGB")

    def get_model(self, layer_level):
        """get VGG_16 varient with only specified blocks

        Args:
            layer_level (int): the block layer that is desired (inclusive)

        Returns:
            keras.model: a VGG_19 model that only contains the specified blocks
        """
        self.__set_layer_weights()

        model = keras.Sequential([], name="VGG19")


        if(layer_level >= 1): # block 1
            model.add(layers.Input((40,40,1)))
            model.add(self.imbedRGB)
            model.add(self.conv1)
        if(layer_level >= 2):
            model.add(self.conv2)
        if(layer_level >= 3):
            model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid', data_format="channels_last", name="block1_pool"))


        if(layer_level >= 4): # block 2
            model.add(self.conv3)
        if(layer_level >= 5):
            model.add(self.conv4)
        if(layer_level >= 6):
            model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid', data_format="channels_last", name="block2_pool"))

        if(layer_level >= 7): # block 3
            model.add(self.conv5)
        if(layer_level >= 8):
            model.add(self.conv6)
        if(layer_level >= 9):
            model.add(self.conv7)
        if(layer_level >= 10):
            model.add(self.conv8)
        if(layer_level >= 11):
            model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid', data_format="channels_last", name="block3_pool"))

        if(layer_level >= 12): # block 4
            model.add(self.conv9)
        if(layer_level >= 13):
            model.add(self.conv10)
        if(layer_level >= 14):
            model.add(self.conv11)
        if(layer_level >= 15):
            model.add(self.conv12)
        if(layer_level >= 16):
            model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid', data_format="channels_last", name="block4_pool"))

        if(layer_level >= 17): # block 5
            model.add(self.conv13)
        if(layer_level >= 18):
            model.add(self.conv14)
        if(layer_level >= 19):
            model.add(self.conv15)
        if(layer_level >= 20):
            model.add(self.conv16)
        if(layer_level >= 21):
            model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid', data_format="channels_last", name="block5_pool"))

        if(layer_level >= 22): # block 6
            model.add(self.flatten)
        if(layer_level >= 23):
            model.add(self.hidden_dense1)
        if(layer_level >= 24):
            model.add(self.hidden_dense2)

        if(self.output and layer_level < 22):
            model.add(self.flatten)
            model.add(self.output_layer)
        return model

            
            

    def __set_layer_weights(self):
        """set the weights of all the layers in the model, from the weigths that were passed
            at instantiation
        """ 
        
        # Convolution Layers
        #----------------------------------------
        # Build The Layers
        self.conv1.build((227,227,3))
        self.conv2.build((1,227,227,64))

        self.conv3.build((1,227,227,64))
        self.conv4.build((1,112,112,128))

        self.conv5.build((1,112,112,128))
        self.conv6.build((1,56,56,256))
        self.conv7.build((1,56,56,256))
        self.conv8.build((1,56,56,256))

        self.conv9.build((1,56,56,256))
        self.conv10.build((1,28,28,512))
        self.conv11.build((1,28,28,512))
        self.conv12.build((1,14,14,512))

        self.conv13.build((1,14,14,512))
        self.conv14.build((1,7,7,512))
        self.conv15.build((1,7,7,512))
        self.conv16.build((1,7,7,512))
        
        # Set the weights
        self.conv1.set_weights([self.wd[0], self.wd[1]])
        self.conv2.set_weights([self.wd[2], self.wd[3]])
        self.conv3.set_weights([self.wd[4], self.wd[5]])
        self.conv4.set_weights([self.wd[6], self.wd[7]])
        self.conv5.set_weights([self.wd[8], self.wd[9]])
        self.conv6.set_weights([self.wd[10], self.wd[11]])
        self.conv7.set_weights([self.wd[12], self.wd[13]])
        self.conv8.set_weights([self.wd[14], self.wd[15]])
        self.conv9.set_weights([self.wd[16], self.wd[17]])
        self.conv10.set_weights([self.wd[18], self.wd[19]])
        self.conv11.set_weights([self.wd[20], self.wd[21]])
        self.conv12.set_weights([self.wd[22], self.wd[23]])
        self.conv13.set_weights([self.wd[24], self.wd[25]])
        self.conv14.set_weights([self.wd[26], self.wd[27]])
        self.conv15.set_weights([self.wd[28], self.wd[29]])
        self.conv16.set_weights([self.wd[30], self.wd[31]])

        # Fully Connected Layers
        #---------------------------------------
        #Build The Layers
        self.hidden_dense1.build((1,25088))
        self.hidden_dense2.build((1,4096))


        # Set the weights
        self.hidden_dense1.set_weights([self.wd[32], self.wd[33]])
        self.hidden_dense2.set_weights([self.wd[34], self.wd[35]])





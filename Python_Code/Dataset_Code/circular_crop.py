"""
File    : rotate_sets.py
Author  : J.Burnham
Date    : 05/19/2022
Org     : FGCU
Purpose : Definition of CircularCropSets class which takes a image set (numpy array),
          and rotates it to the given degree
"""


import threading
from tqdm import tqdm
import numpy as np
from PIL import Image
import scipy.io
from scipy import ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.preprocessing import MinMaxScaler


class CircularCropSets(threading.Thread):
    def __init__(self, image_set, height, width, set_name, degree):
        """
        default init method

        @param image_set:   - the image set to rotate
        @param degree:      - the degree to rotate the image to
        @param set_name:    - the name of the set (will show up on the console window
                               progress bar)

        """
        threading.Thread.__init__(self)
        self.image_set = image_set
        self.height = height
        self.width = width
        self.set_name = set_name
        self.cropped_set = np.zeros(shape=(0, 0, 0))
        self.degree = degree

    def run(self):
        """
        when callin start() on a thread, this is the method is calls
        """
        cropped_data = self.crop()
        self.cropped_set = cropped_data

    def crop(self):
        """
        rotate the image set to the desired degree

        @returns: numpy array   - the resized set
        """
        new_image_set = np.zeros(shape=(0,self.height, self.width))

        for i in range(self.image_set.shape[0]):
            image = self.image_set[i] #load 
            image = np.reshape(image, (40,40))

            image = Image.fromarray(image) #resize
            image = image.resize((self.height,self.width))
            image = np.array(image)
            
            image -= np.mean(image)
            image /= np.abs(image).max()


            image = ndimage.rotate(image, self.degree, reshape=False, prefilter=False, order = 0)


            # multiply by the cicular mask that diMattina set you here
            mask = scipy.io.loadmat("TaperMask.mat")['Imask']

            for y in range(len(image)):
                for x in range(len(image[0])):
                    pixel = image[y][x] * mask[y][x]

                    image[y][x] = pixel


            image = np.expand_dims(image, axis=0)
            new_image_set = np.append(new_image_set, image, axis=0)

        return new_image_set

    

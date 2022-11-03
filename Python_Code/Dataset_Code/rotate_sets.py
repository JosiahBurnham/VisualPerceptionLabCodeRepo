"""
File    : rotate_sets.py
Author  : J.Burnham
Date    : 05/19/2022
Org     : FGCU
Purpose : Definition of RotateSets class which takes a image set (numpy array),
          and rotates it to the given degree
"""


import numpy as np
from scipy import ndimage
import threading
from tqdm import tqdm


class RotateSets(threading.Thread):
    def __init__(self, image_set, degree, set_name):
        """
        default init method

        @param image_set:   - the image set to rotate
        @param degree:      - the degree to rotate the image to
        @param set_name:    - the name of the set (will show up on the console window
                               progress bar)

        """
        threading.Thread.__init__(self)
        self.image_set = image_set
        self.degree = degree
        self.set_name = set_name
        self.rotated_set = np.zeros(shape=(0, 0, 0))

    def run(self):
        """
        when callin start() on a thread, this is the method is calls
        """
        rotated_data = self.resize()
        self.rotated_set = rotated_data

    def resize(self):
        """
        rotate the image set to the desired degree

        @returns: numpy array   - the resized set
        """
        rotated_set = np.zeros(
            shape=(0, self.image_set.shape[1], self.image_set.shape[2]))
        for i in tqdm(range(self.image_set.shape[0]), desc=self.set_name, ascii=True):
            image = self.image_set[i]
            res = ndimage.rotate(image, self.degree, reshape=False)

            res = np.expand_dims(res, axis=0)
            rotated_set = np.append(rotated_set, res, axis=0)
        return rotated_set

"""
File    : resize_sets.py
Author  : J.Burnham
Date    : 01/03/2022
Org     : FGCU
Purpose : Definition of ResizeSets class which takes a image set (numpy array),
          and resizes it to the given dimensions
"""

import numpy as np
from skimage.transform import resize
import threading
from tqdm import tqdm


class ResizeSets(threading.Thread):
    def __init__(self, image_set, x_rez, y_rez, set_name):
        """
        default init method

        @param image_set:   - the image set to resize
        @param x_rez:       - the desired x resolution
        @param y_rez:       - the desired y resolution
        @param set_name:    - the name of the set (will show up on the console window
                               progress bar)

        """
        threading.Thread.__init__(self)
        self.image_set = image_set
        self.x_rez = x_rez
        self.y_rez = y_rez
        self.set_name = set_name
        self.resized_set = np.zeros(shape=(0, 0, 0))

    def run(self):
        """
        when callin start() on a thread, this is the method is calls
        """
        resized_data = self.resize()
        self.resized_set = resized_data

    def resize(self):
        """
        resizes the image set to the desired demensions

        @returns: numpy array   - the resized set
        """
        resized_set = np.zeros(shape=(0, self.x_rez, self.y_rez))
        for i in tqdm(range(self.image_set.shape[0]), desc=self.set_name, ascii=True):
            image = self.image_set[i]
            res = resize(image, (self.x_rez, self.y_rez))
            res = np.expand_dims(res, axis=0)
            resized_set = np.append(resized_set, res, axis=0)
        return resized_set

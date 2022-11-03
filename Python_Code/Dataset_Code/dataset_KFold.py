"""
File:   filterbank_maker.py
Author: Josiah Burnham (10/2021)
Org:    FGCU
Desc:   Take .mat file dataset batches, and return data, in two large tensors (train, test) specifying the hold back set
"""

import numpy as np
import scipy.io
from sklearn.model_selection import KFold


class KFoldDataset:
    """
    preform K-Fold crossvalidation on your data
    """

    # intitializer(s)
    #------------------------------------------------------
    def __init__(self,data, num_folds=10):
        """
        :param file_path:   root file path
        :param num_files:   number of data files
        :param num_img:     number of images per file
        :param x_shape:     the shape of the image height * image width
        :param y_shape:     the number of classifications possible
        :param num_folds:   the number of folds to perform on the database
        """
        self.data = data
        self.num_folds = num_folds
        self.x = self.data[0]
        self.y = self.data[1]


    # member methods
    #------------------------------------------------------

    # public methods
    def make_folds(self):
        """
        the main handler for the dataset maker class

        :returns:       A 3D tensor [num_folds, num_images, image_height * image_width]
        """

        (x_train, y_train), (x_test, y_test) = self.__k_fold(self.x, self.y)

        return (x_train, y_train), (x_test, y_test)




    def __k_fold(self, dataset_x, dataset_y):
        """
        splits the train and test datasets into x and y batches

        :param dataset_x:       The images dataset
        :param dataset_y:       the labels dataset
        :returns: returns two tuples separated into (x_train, y_train) (x_test, y_test)
        """

        batch_size = self.x.shape[0]

        x_train_shape = (batch_size) - ((batch_size) // self.num_folds)
        x_test_shape = (batch_size) // self.num_folds

        x_train_indices, x_test_indices = self.__split_dataset(dataset_x, x_train_shape, x_test_shape)
        y_train_indices, y_test_indices = self.__split_dataset(dataset_y, x_train_shape, x_test_shape)

        x_train_batches = dataset_x[x_train_indices]
        x_test_batches = dataset_x[x_test_indices]

        y_train_batches = dataset_y[y_train_indices]
        y_test_indices = dataset_y[y_test_indices]

        return (x_train_batches, y_train_batches), (x_test_batches, y_test_indices)


    def __split_dataset(self, dataset, train_shape, test_shape):
        """
        splits the dataset into test and train sets, and performs k fold cross validation

        :returns:   two 3D tensors [num_folds, number of test or train  images, img_height * img_width]
        """
        k_fold = KFold(n_splits=self.num_folds)

        train = np.ndarray(shape=(0, train_shape), dtype=np.int32)
        test = np.ndarray(shape=(0, test_shape), dtype=np.int32)

        for train_indices, test_indices in k_fold.split(dataset):
            train_indices = np.expand_dims(train_indices, axis=0)
            test_indices = np.expand_dims(test_indices, axis=0)

            test = np.append(test_indices, test, axis=0)
            train = np.append(train_indices, train, axis=0)
        return train, test


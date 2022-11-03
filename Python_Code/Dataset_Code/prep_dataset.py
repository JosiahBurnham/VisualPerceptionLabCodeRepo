"""
File:   prep_dataset.py
Author: Josiah Burnham (10/2021)
Org:    FGCU
Desc:   Prepare dataset for training
"""

import numpy as np


class PrepDataset:
    """
    Preps the data for either the beginning of training or for each step in the train loop
    """

    # initializer(s)
    #------------------------------------------------------
    def __init__(self, x_train, y_train, x_test, y_test, test_fold):
        """
        Standard constructor for the PrepDataset class

        @param x_train:     the features of the training dataset
        @param y_train:     the labels for the training set
        @param x_test:      the features of the test dataset
        @param y_test:      the labels for the test dataset
        @param test_fold:   the fold of the total dataset we are training on
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.test_fold = test_fold

    # member methods
    #======================================================

    # public methods
    #------------------------------------------------------
    def prep_epoch_data(self, do_shuffle = False):
        """
        Formats and Shuffles each different subset of the total dataset to begin the training process

        @return: (x_train_batch, y_train_batch), (x_test_batch, y_test_batch)
                 The prepped batches for each subset of the total dataset to being the training process on
        """
        # choosing which fold to test on, to train on the rest (K-Fold cross validation)
        x_train_batch = self.x_train[self.test_fold]
        y_train_batch = self.y_train[self.test_fold]

        # making all the images into a 3D tensor to eventually pass to the convolution function
        x_train_batch = np.reshape(x_train_batch, newshape=((x_train_batch.shape[0],40, 40)))

        if do_shuffle:
            x_train_batch, y_train_batch = self.__shuffle_data(x_train_batch, y_train_batch, x_train_batch.shape[0])

        x_test_batch = self.x_test[self.test_fold]
        y_test_batch = self.y_test[self.test_fold]

        x_test_batch = np.reshape(x_test_batch, newshape=((x_test_batch.shape[0],40,40)))

        return (x_train_batch, y_train_batch), (x_test_batch, y_test_batch)


    def prep_step_data(self, batch_size, step):
        """
        prepare and shuffle the data used for inside of a training step

        @param batch_size: the batch size used for training
        @param step:       the current step of the training loop
        @return: (x_step_train, y_step_train), (x_step_test, y_step_test)
                 The prepped batches for each subset of the total Batch the loop is currently training on
        """
        train_indices = []
        for i in range(batch_size):
            train_indices.append(i + (step * batch_size))

        x_step_train, y_step_train = self.__shuffle_step_data(self.x_train, self.y_train,train_indices, batch_size)

        x_step_test, y_step_test = self.__shuffle_step_data(self.x_test, self.y_test, self.x_test.shape[0], batch_size)

        return (x_step_train, y_step_train), (x_step_test, y_step_test)


    # private methods
    #------------------------------------------------------
    @staticmethod
    def __shuffle_data(x_data, y_data, range):
        """
        Shuffle the features and the labels of a dataset within a range

        @param x_data: the features of a dataset
        @param y_data: the labels of a dataset
        @param range:  the range of values for the rand function
        @return: the shuffled data for x and y
        """
        shuffle_var = np.random.permutation(range)
        x_data = x_data[shuffle_var, :]
        y_data = y_data[shuffle_var, :]

        return x_data, y_data


    @staticmethod
    def __shuffle_step_data(x_data, y_data, indices, batch_size):
        """
        Shuffle the features and labels used in the training step

        @param x_data: the features of a dataset
        @param y_data: the labels of a dataset
        @param indices: the range of indices the rand function can choose from
        @param batch_size: the batch sized used in the training loop
        @return: the shuffled data for x and y
        """
        shuffle_var = np.random.permutation(indices)
        shuffle_var = shuffle_var[0: batch_size]

        x_step_data = x_data[shuffle_var, :]
        y_step_data = y_data[shuffle_var, :]

        return x_step_data, y_step_data

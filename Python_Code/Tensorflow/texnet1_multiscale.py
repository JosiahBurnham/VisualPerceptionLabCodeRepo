# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 10:45:27 2019

@author: cdimattina

This program defines a convolutional neural network which is applied to 
classify natural textures and texture boundaries into categories

Training set naming conventions
-------------------------------
train_set_{numclasses}_{setnum}.mat

Requires
--------
filter_mat_multiscale1.mat : This program contains a set of 32x32 filters

Usage
-----
To load the module : "import texnet1_multiscale as tx1"
To train a network : "tx1.train_model()" - this will use default settings
    
"""

###############################################################################
# Import packages needed 
###############################################################################
from __future__ import division, print_function
import tensorflow as tf
import numpy as np
import scipy.io
from scipy.io import savemat

###############################################################################
# Define essential constants : NETWORK / DATA
###############################################################################
image_dim = 64
num_input = image_dim * image_dim
num_classes = 10
v2_stride = 8
v4_stride = 3

# TRAINING CONSTANTS 
train_iters = 50
num_cycles = 10
batch_size = 100
display_step = 10
learning_rate = 0.01

###############################################################################
# Define network and intialize weights
###############################################################################

tf.compat.v1.disable_eager_execution()

def init_weights(filter_str, num_v2, num_v4):
    global weights
    global biases

    # Load fixed filters
    fixed_filters = scipy.io.loadmat(filter_str)['filter_mat']
    fixed_filter_size = np.size(fixed_filters, 0)
    nfixed_filters = np.size(fixed_filters, 3)

    # Print announcement
    print("...fixed filters loaded...Filters = " + str(nfixed_filters) +
          ", Size = " + str(fixed_filter_size) + "...")

    weights = {
        # 32x32 convolutional filters, 1 input, 6 outputs
        'wc1': tf.Variable(fixed_filters, dtype=tf.float32, trainable=False),
        # 4x4   convolutional filters, 6 input, num_v2 outputs
        'wc2': tf.Variable(0.1 * tf.random.normal([v4_stride, v4_stride, nfixed_filters, num_v2]), dtype=tf.float32),
        # Fully connected: 2 outputs (L/R)
        'wd1': tf.Variable(0.1 * tf.random.normal([v4_stride * v4_stride * num_v2, num_v4]), dtype=tf.float32),
        'wd2': tf.Variable(0.1 * tf.random.normal([num_v4, num_classes]), dtype=tf.float32)
    }

    biases = {
        # 6 filters
        'bc1': tf.Variable(tf.zeros([nfixed_filters]), dtype=tf.float32, trainable=False),
        # num_v2 filters
        'bc2': tf.Variable(tf.zeros([num_v2]), dtype=tf.float32, trainable=True),
        # 2 units
        'bd1': tf.Variable(tf.zeros([num_v4]), dtype=tf.float32, trainable=True),
        'bd2': tf.Variable(tf.zeros([num_classes]), dtype=tf.float32, trainable=True)
    }

    print("...weights initialized...")


###############################################################################
# Define network operations
###############################################################################

def conv2dabs(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.abs(x)


def conv2drelu(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def avgpool2d(x, k):
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def conv_net(x, weights, biases):
    x = tf.reshape(x, shape=[-1, image_dim, image_dim, 1])

    # First convolutional layer, with full-wave rectification
    conv1 = conv2drelu(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=v2_stride)

    # Second convolutional layer, with half-wave rectification
    conv2 = conv2drelu(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=v4_stride)

    # Reshape input to match fully connected layer
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])

    # Fully connected layer: Different texture categories
    fc1 = tf.abs(tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1']))
    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    out = fc2

    return out, conv2, fc1


###############################################################################
# Define TensorFlow graph
###############################################################################

def define_graph(l2penalty, sparse_penalty, learning_rate):
    global X, y, keep_prob
    global pred, errorterm, weight_norm, cost, optimizer, correct_prediction, accuracy

    # This placeholder will hold the training batch
    X = tf.compat.v1.placeholder("float", [100, 15])
    y = tf.compat.v1.placeholder("float", [100, 15])
    keep_prob = tf.compat.v1.placeholder(tf.float32)

    pred, conv2, fc1 = conv_net(X, weights, biases)
    errorterm = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    weight_norm = tf.reduce_mean(tf.pow(weights['wd1'], 2)) + tf.reduce_mean(tf.pow(weights['wc2'], 2))
    sparse_term = tf.reduce_mean(tf.abs(conv2)) + tf.reduce_mean(tf.abs(fc1))
    cost = errorterm + l2penalty * weight_norm + sparse_penalty * sparse_term
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def train_model(fname="train_set_text_onescale_idl", num_files=6, num_v2=6, num_v4=6, testfile=5, l2penalty=0.0,
                sparse_penalty=0.0):
    # Initialize network weights: Global varibles <weights>, <biases>
    filter_str = 'filter_mat_onescale.mat'
    init_weights(filter_str, num_v2, num_v4)

    # General strategy: We do not want to over-load memory, so we load training files
    # and then train on them for a while. We then load the next file and train on it, 
    # and so on and so forth, until we have cycled through all training files. We hold
    # back one file as the test file - it is not used to train, but only to evaluate.     
    train_list = [i for i in range(num_files)]

    # Remove the test files from the training set
    train_list.remove(testfile)

    # Define vectors for plotting loss function
    train_loss = []  # loss function
    train_acc = []  # accuracy on training set
    train_err = []  # error term without l2 penalty - train

    test_loss = []  # loss for test set
    test_acc = []  # accuracy on test set
    test_err = []  # error term without l2 penalty - test
    test_errhp = []  # error term without l2 penalty - hyper-parameter opt

    # Print the outfilename
    matfile_name = "C:\\Users\\jjburnham0705\\Desktop\\PYTHON\MODELS\\" + fname + "_" + str(num_v2) + "_" + str(
        num_v4) + "_" + str(testfile) + "_" + str(l2penalty) + "_" + str(sparse_penalty)

    print("...output file: " + matfile_name)

    # Start a new TF session
    with tf.compat.v1.Session() as sess:

        # Set up TF graph
        define_graph(l2penalty, sparse_penalty, learning_rate)
        print("...tensorflow graph defined...")

        # Run the initializer
        init = tf.compat.v1.global_variables_initializer()
        print("...tensorflow global variables initialized...")
        sess.run(init)
        print("...tensorflow session started...")

        # Load the test set
        fname_test = "C:\\Users\\jjburnham0705\\Desktop\\PYTHON\\TRAINDATA\\" + fname + "_" + str(testfile) + ".mat"

        test_batch_all_x = scipy.io.loadmat(fname_test)['Imat']
        test_batch_all_y = scipy.io.loadmat(fname_test)['Rt']

        num_stim_test_batch = np.shape(test_batch_all_y)[0]
        num_stim_test_batch2 = int(num_stim_test_batch / 2)

        # test batch for evaluating accurcay
        test_batch_x = test_batch_all_x[0:num_stim_test_batch2, :]
        test_batch_y = test_batch_all_y[0:num_stim_test_batch2, :]

        # second test batch for hyper-parameter optimization
        test_batch_x_hp = test_batch_all_x[num_stim_test_batch2:num_stim_test_batch, :]
        test_batch_y_hp = test_batch_all_y[num_stim_test_batch2:num_stim_test_batch, :]

        # Cycle through specified number of cycles
        for cur_cycle in range(num_cycles):
            for cur_trainset in train_list:
                fname_train = "./TRAINDATA/" + fname + "_" + str(cur_trainset) + ".mat"

                train_batch_all_x = scipy.io.loadmat(fname_train)['Imat']
                train_batch_all_y = scipy.io.loadmat(fname_train)['Rt']

                num_stim = np.shape(train_batch_all_x)[0]

                step = 1
                while step < train_iters:
                    perm_temp = np.random.permutation(num_stim)
                    perm_temp = perm_temp[0:batch_size]

                    train_batch_x = train_batch_all_x[perm_temp, :]
                    train_batch_y = train_batch_all_y[perm_temp, :]

                    print(train_batch_x.shape)

                    print({X:train_batch_x, y:train_batch_y})

                    sess.run(optimizer, feed_dict={X: train_batch_x, y: train_batch_y})
                    step += 1

                    if step % display_step == 0:
                        # Calculate accuracy for training images
                        loss_train, acc_train, err_train = sess.run([cost, accuracy, errorterm],
                                                                    feed_dict={X: train_batch_x, y: train_batch_y})

                        # Calculate accuracy for test images
                        loss_test, acc_test, err_test = sess.run([cost, accuracy, errorterm],
                                                                 feed_dict={X: test_batch_x, y: test_batch_y})

                        # Calculate goodness of fit for hyper-parameter optimization test batch
                        err_test_hp = sess.run([errorterm], feed_dict={X: test_batch_x_hp, y: test_batch_y_hp})

                        print("Cycle " + str(cur_cycle) + ", Train Set " + str(cur_trainset) + ", Iter " + str(step) + \
                              ", Minibatch Loss= " + "{:.2f}".format(loss_train) \
                              + ", Minibatch Error= " + "{:.2f}".format(err_train) \
                              + ", Test Error= " + "{:.2f}".format(err_test) \
                              + ", Training Accuracy= " + "{:.2f}".format(acc_train) \
                              + ", Testing Accuracy= " + "{:.2f}".format(acc_test))

                        # Append current measurements to vectors
                        train_loss.append(loss_train)
                        train_acc.append(acc_train)
                        train_err.append(err_train)

                        test_loss.append(loss_test)
                        test_acc.append(acc_test)
                        test_err.append(err_test)
                        test_errhp.append(err_test_hp)

                # end while
            # end for
        # end for

        # Get predictions of the final trained model for all the stimuli in the test set
        net_pred, dum1, dum2 = conv_net(test_batch_x, weights, biases)
        test_batch_ypred = tf.nn.softmax(net_pred)

        # Save everything to a .mat file in the output directory!
        wc1_final = weights['wc1'].eval()
        wc2_final = weights['wc2'].eval()
        wd1_final = weights['wd1'].eval()
        wd2_final = weights['wd2'].eval()

        bc1_final = biases['bc1'].eval()
        bc2_final = biases['bc2'].eval()
        bd1_final = biases['bd1'].eval()
        bd2_final = biases['bd2'].eval()

        test_pred = test_batch_ypred.eval();

        mydict = {"wc1": wc1_final, "wc2": wc2_final, "wd1": wd1_final, \
                  "bc1": bc1_final, "bc2": bc2_final, "bd1": bd1_final, \
                  "wd2": wd2_final, "bd2": bd2_final, \
                  "train_loss": train_loss, \
                  "train_acc": train_acc, "test_loss": test_loss, "test_acc": test_acc, \
                  "train_err": train_err, "test_err": test_err, "test_errhp": test_errhp, \
                  "test_pred": test_pred, \
                  "l2penalty": l2penalty, "testfile": testfile}

        savemat(matfile_name, mydict)


# Print announcement that module is loaded
print("...module imported...")

train_model()

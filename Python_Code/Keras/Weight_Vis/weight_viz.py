"""
File    : weight_vis.py
Author  : J.Burnham (9/2021)
Purpose : Visualize the weights of a train convolutional neural network
"""


import numpy as np
import tensorflow as tf
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.colors

def main():

    loaded_weights = load_weights("Models\\one_SHAD_EDGE_GRY_40x40_18_10.0.mat", 1)

    filter_weights = np.zeros((5, 5, 0))
    iters = 0

    for i in range(104):
    # Num Filters

        img = np.zeros((0,5))
        for height in range(5):

        # Image Height
            np_img_row = np.zeros(0)
            img_row = []
            img_row.clear()

            for width in range(5):
                # Image Width
                weight = loaded_weights[iters]
                img_row.append(weight)

                iters += 1

            np_img_row = np.append(np_img_row, img_row, axis=0)
            np_img_row = tf.expand_dims(np_img_row, axis=0)

            img = np.append(img, np_img_row, axis=0)
        img = tf.expand_dims(img, axis=2)
        filter_weights = np.append(filter_weights, img, axis=2)

    fig = plt.figure(figsize=(10, 5))
    fig.suptitle("4x4 Spatial Scale Filters:")
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["orangered", "darkkhaki", "royalblue"], N=8)

    ax = plt.axes([0.92, 0.1, 0.015, 0.8])

    cbar = fig.colorbar(plt.cm.ScalarMappable( cmap=cmap), cax=ax)

    iters = 0
    for file in  range(52, 78):
        img = filter_weights[:, :, file]
        fig.add_subplot(3, 9, iters+1)
        plt.imshow(img, cmap= cmap)
        plt.axis("off")
        plt.title("Filter: " + str(iters+1))
        iters +=1


    plt.show()


def load_weights(file_path,weight_type):
    weights = scipy.io.loadmat(file_path)["wd"]
    selected_weight = weights[:, weight_type]

    return selected_weight

if __name__ == "__main__":
    main()
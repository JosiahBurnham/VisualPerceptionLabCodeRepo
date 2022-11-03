"""
File    : weight_vis_r.py
Author  : J.Burnham (10/2021)
Purpose : Visualize the weights of a trained convolutional neural network
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.colors


def main():
    loaded_weights = load_weights(
        "Models\\one_SHAD_EDGE_GRY_40x40_18_10.0.mat", 0)

    len_of_weights = loaded_weights.shape[0]
    filter_weights = np.zeros(shape=(0, 104))

    # make 2D tensor
    for i in range(25):

        row_weights = np.zeros(shape=(0))
        for j in range(104):
            row_weights = np.append(
                row_weights, loaded_weights[len_of_weights - 1])
            len_of_weights -= 1
        row_weights = np.expand_dims(row_weights, axis=0)
        filter_weights = np.append(filter_weights, row_weights, axis=0)

    # make 3D tensor
    weights = np.zeros(shape=(5, 5, 0))
    for i in range(104):
        image = np.zeros(shape=(5, 0))
        image_raw = filter_weights[:, i]

        len_of_pixels = filter_weights.shape[0]
        for j in range(5):
            image_width = np.zeros(shape=(0))
            for k in range(5):
                image_width = np.append(
                    image_width,  image_raw[len_of_pixels-1])
                len_of_pixels -= 1

            image_width = np.expand_dims(image_width, axis=1)
            image = np.append(image, image_width, axis=1)
        image = np.expand_dims(image, axis=2)
        weights = np.append(weights, image, axis=2)

    # plot the weights as an image
    fig = plt.figure(figsize=(10, 6), tight_layout=True)
    fig.suptitle("Edge Category: 32x32 Spatial Scale Filter Weights")
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "", ["orangered", "darkkhaki", "royalblue"], N=4)
    ax = plt.axes([0.1, 0.05, 0.8, 0.03])

    cbar = fig.colorbar(plt.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=-np.max(loaded_weights), vmax=np.max(loaded_weights)), cmap=cmap), cax=ax, orientation="horizontal")

    iters = 0
    for file in range(26, 52):
        img = weights[:, :, file]
        fig.add_subplot(3, 9, iters + 1)
        plt.imshow(img, cmap=cmap)
        plt.clim(np.min(loaded_weights), np.max(loaded_weights))

        plt.axis("off")
        plt.title("Filter: " + str(iters + 1))
        iters += 1

    plt.show()


def load_weights(file_path, weight_type):
    weights = scipy.io.loadmat(file_path)["wd"]
    selected_weight = weights[:, weight_type]

    return selected_weight


if __name__ == "__main__":
    main()

"""
file:   pick_filter.py
by:     Josiah Burnham
org:    FGCU
desc:   loads an image, picks a filter based on input, filters the image, and save the output.
"""
import tensorflow as tf
import tensorflow.keras.preprocessing as preprocessing
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from fixed_filter_layer import FilterLayer


def splash_screen():
    """
    Prints a splash screen to greet the user
    """
    print("------------------------------------------------------------------")
    print("    _____ _      _                       ______ _ _ _")
    print("   |  __ (_)    | |           /\        |  ____(_) | |")
    print("   | |__) |  ___| |          /  \       | |__   _| | |_ ___ _ __")
    print("   |  ___/ |/ __| |/ /      / /\ \      |  __| | | | __/ _ \ '__|")
    print("   | |   | | (__|   <      / ____ \     | |    | | | ||  __/ |")
    print("   |_|   |_|\___|_|\_\    /_/    \_\    |_|    |_|_|\__\___|_| ")
    print("------------------------------------------------------------------")


def load_img(file_path):
    """
    Loads an Image using the preprocessing library within the keras API

    :param file_path: the file path of the image to load
    :return: returns a PIL instance of the image
    """

    og_img = preprocessing.image.load_img(file_path)
    return og_img


def preprocess_image(og_img):
    """
    Preforms basic preprocessing on the image before filtering.
    1) converts to image to float32 datatype,
    2) converts image to greyscale,
    3) normalizes the image

    :param og_img: the image to preprocess
    :return: returns the preprocessed image
    """

    img = np.float32(og_img)  # convert uint8 into float32
    img = tf.image.rgb_to_grayscale(img)
    img = img / 255.0  # normalize the image
    return img


def load_filters(filter_path, filter_name):
    """
    Loads pre-saved filters from a .mat file

    :param filter_path: path to the .mat file
    :param filter_name: the name of the filter(s) in the file

    :return: returns the filter(s) in a numpy array
    """
    return scipy.io.loadmat(filter_path)[filter_name]


def get_input():

    print("\n\nWhat size filter would you like to use? (32,16,8,4):", end=" ")
    filter_size = input()
    print("\n Which filter would you like to pick in that scale? (1-26):", end=" ")
    filter_choice = int(input())
    return filter_size, filter_choice-1


def filter_image(loaded_fixed_filters, filter_choice, expanded_img):
    """
    Filters the image passed with the user selected image

    :param loaded_fixed_filters: the numpy array of all the filters in the .mat file
    :param filter_choice: the filter the user choice (an int from 1-26)
    :param expanded_img: the image with a fourth dimension for batch sized added to the beginning of the image  matrix

    :return: the filtered image, and selected filter as numpy arrays to display with matplotlib
    """
    selected_filter = loaded_fixed_filters[:, :, :, filter_choice]  # singling out the filter to use

    # creating a four dimensional tensor of the filter to pass into the FilterLayer class
    fixed_filter = tf.Variable(selected_filter, dtype=tf.float32, trainable=False)

    filter_layer = FilterLayer(fixed_filter)

    # the convolution method requires a four dimensional tensor which is why the expanded image is used
    filtered_img = filter_layer(expanded_img)

    # getting rid of the batch dimension to display the image with matplotlib
    filtered_img = tf.squeeze(filtered_img, axis=0)

    return filtered_img, selected_filter


def make_figure(og_img, img, selected_filter, filtered_img):
    """
    displays the four different images that signifies the steps this program goes through and saves it as a PNG

    :param og_img: the original image that was loaded
    :param img: the image with pre processing done to it
    :param selected_filter: the filter that the user selected
    :param filtered_img:  the image convolved with the selected filter

    :return: None
    """

    # creates a figure
    fig = plt.figure(figsize=(10, 7))

    # sets background color
    fig.set_facecolor((0, 0, 0))

    # adds a subplot in the figure where the image will be
    fig.add_subplot(2, 2, 1)

    plt.imshow(og_img)
    plt.axis("off")
    plt.title("Original Image", color="white")

    fig.add_subplot(2, 2, 2)

    plt.imshow(img, cmap="Greys")
    plt.axis("off")
    plt.title("Pre-Processed Image", color="white")

    fig.add_subplot(2, 2, 3)

    plt.imshow(selected_filter)
    plt.axis("off")
    plt.title("Selected Filter", color="white")

    fig.add_subplot(2, 2, 4)

    plt.imshow(filtered_img, cmap="Greys")
    plt.axis("off")
    plt.title("Filtered Image", color="white")

    # saves the figure
    plt.savefig("filters.png")
    plt.show()


def main():
    """
    Functions as the main switchboard for this program.
    :return: None
    """

    og_img = load_img("test_pic.jpg")

    processed_img = preprocess_image(og_img)

    # adding batch size to the image (recommended by the tf.nn.conv2D docs)
    expanded_img = tf.expand_dims(processed_img, axis=0)

    splash_screen()

    is_ending = False

    while(is_ending == False):

        filter_size,filter_choice = get_input()

        loaded_fixed_filters = load_filters("first_stage_"+filter_size+".mat", "filter_mat")

        filtered_img, selected_filter = filter_image(loaded_fixed_filters, filter_choice, expanded_img)

        make_figure(og_img, processed_img, selected_filter, filtered_img)

        print("Would you like to continue (Y/N):", end=" ")
        choice = input()
        if(choice.lower() == "n"):
            is_ending = True


main()

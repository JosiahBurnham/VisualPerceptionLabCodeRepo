"""
File    : main.py
Author  : J.Burnham
Date    : 05/20/2022
Org     : FGCU
Purpose : main application file for the Circular Crop Image Sets Program
"""

import scipy
import sys
import getopt
import numpy as np
import matplotlib.pyplot as plt

from Scripts.load_data import LoadDataset
from Scripts.circular_crop import CircularCropSets


def crop_handler(file_path, height, width, degree):
    """
    Handles the threading of the task of circularly cropping datasets that are stored in .mat files with
    dict labels that are ["image_patches", "category_labels"]. this method uses four threads
    to optimize this task. it then joins the thread data back together, and outputs the cropped
    data to another .mat file with the same dict labels

    @param file_path:   - the file path to the data dir where the orginal data is stored
    @param num_files:   - the number of files in the directory that is to be rotated
    @param height       - the height of the circular crop window
    @param width        - the width of the circular crop window
    """

    x_data_sets = []
    y_data_sets = []

    start_file = 1  # the first file in the data dir should be labeled with a one on the end

    loadData = LoadDataset(dir_path=file_path, x_shape=1600, isSplit=False)
    (x_data, y_data) = loadData.load_dataset()
    print(x_data.shape)
    print(y_data.shape)

    for i in range(4):
        upperbound = (x_data.shape[0] // 4) * \
            i + (x_data.shape[0] // 4)

        lowerbound = (x_data.shape[0] // 4) * i

        x_data_sets.append(x_data[lowerbound:upperbound])
        y_data_sets.append(y_data)

        i += 1

    
    # define all threads
    first_data_thread = CircularCropSets(
        x_data_sets[0], height, width, "Subset 1", degree)
    second_data_thread = CircularCropSets(
        x_data_sets[1], height, width, "Subset 2", degree)
    third_data_thread = CircularCropSets(
        x_data_sets[2], height, width, "Subset 3", degree)
    fourth_data_thread = CircularCropSets(
        x_data_sets[3], height, width, "Subset 4", degree)
        
    # start all threads
    first_data_thread.start()
    second_data_thread.start()
    third_data_thread.start()
    fourth_data_thread.start()

    # wait for all threads to finish
    first_data_thread.join()
    second_data_thread.join()
    third_data_thread.join()
    fourth_data_thread.join()

    # rejoin the resized sets
    fully_cropped_set = np.concatenate((first_data_thread.cropped_set, second_data_thread.cropped_set,
                                       third_data_thread.cropped_set, fourth_data_thread.cropped_set))

    new_y_data = np.zeros(
    shape=(0, 2), dtype=int)

    for i in range(fully_cropped_set.shape[0]):
        data = np.zeros(
            shape=(1, 2), dtype=int
        )

        if degree == 45:
            #right
            data[0][0] = 0
            data[0][1] = 1
        else:
            #left
            data[0][0] = 1
            data[0][1] = 0
        new_y_data = np.append(new_y_data, data, axis=0)


    # Save cropped Data set to a .mat file
    matfile_name = "./Circularly_Cropped_Sets/" + "Cropped_Set" + \
        "_" + str(fully_cropped_set.shape) +"_"+ file_path[-1] + ".mat"
    save_dict = {"image_patches": fully_cropped_set,
                 "category_labels": new_y_data}
    try:
        scipy.io.savemat(matfile_name, save_dict)
    except FileNotFoundError:
        f = open(matfile_name, "x")
        f.close()
        scipy.io.savemat(matfile_name, save_dict)

def main(argv):
    """
    gets the arguments passed from the command line, then passes those args to the rotate method
    @param argv: The arguments from the CLI
    """
    file_path = ""
    height = 0
    width = 0
    degree = 0

    try:
        opts, args = getopt.getopt(
            argv,
            "h:f:t:w:d:",
            ["file_path=", "height=", "width=", "degree="],
        )
    except getopt.GetoptError:
        print(
            "\n Invalid Syntax: main.py -f <file path to train dir> -n <num of files in dir> -h <height of crop window> -w <width of crop window>\n"
        )
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(
                "\nSyntax: main.py -f <file path to train dir> -n <num of files in dir> -h <height of crop window> -w <width of crop window>\n"
            )
        elif opt in ("-f", "--file_path"):
            file_path = arg
        elif opt in ("-t", "--height"):
            height = int(arg)
        elif opt in ("-w", "--width"):
            width = int(arg)
        elif opt in ("-d", "--degree"):
            degree = int(arg)

    crop_handler(file_path, height, width, degree)


if __name__ == "__main__":
    main(sys.argv[1:])

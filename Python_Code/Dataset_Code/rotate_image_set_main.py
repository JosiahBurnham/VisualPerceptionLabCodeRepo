"""
File    : main.py
Author  : J.Burnham
Date    : 05/18/2022
Org     : FGCU
Purpose : main application file for the Rotate Image Sets Program
"""

import scipy
import sys
import getopt
import numpy as np

from Scripts.load_data import LoadData
from Scripts.rotate_sets import RotateSets


def rotate_handler(file_path, num_files, degree):
    """
    Handles the threading of the task of rotating datasets that are stored in .mat files with
    dict labels that are ["image_patches", "category_labels"]. this method uses four threads
    to optimize this task. it then joins the thread data back together, and outputs the rotated
    data to another .mat file with the same dict labels

    @param file_path:   - the file path to the data dir where the orginal data is stored
    @param num_files:   - the number of files in the directory that is to be rotated
    @param degree       - the degree to rotate the image to
    """

    x_data_sets = []
    y_data_sets = []

    start_file = 1  # the first file in the data dir should be labeled with a one on the end

    loadData = LoadData(file_path=file_path,
                        num_files=num_files, start_file=start_file)
    (x_data, y_data) = loadData.load_file_data()

    for i in range(4):
        upperbound = (x_data.shape[0] // 4) * \
            i + (x_data.shape[0] // 4)

        lowerbound = (x_data.shape[0] // 4) * i

        x_data_sets.append(x_data[lowerbound:upperbound])
        y_data_sets.append(y_data)

        i += 1

    # define all threads
    first_data_thread = RotateSets(x_data_sets[0], degree, "Subset 1")
    second_data_thread = RotateSets(x_data_sets[1], degree, "Subset 2")
    third_data_thread = RotateSets(x_data_sets[2], degree, "Subset 3")
    fourth_data_thread = RotateSets(x_data_sets[3], degree, "Subset 4")

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
    fully_rotated_set = np.concatenate((first_data_thread.rotated_set, second_data_thread.rotated_set,
                                       third_data_thread.rotated_set, fourth_data_thread.rotated_set))

    all_y_data = np.concatenate(
        (y_data_sets[0], y_data_sets[1], y_data_sets[2], y_data_sets[3]))

    # Save Resized Data set to a .mat file
    matfile_name = "./Rotated_Sets/" + "rotated_set" + \
        "_" + str(fully_rotated_set.shape) + ".mat"
    save_dict = {"image_patches": fully_rotated_set,
                 "category_labels": all_y_data}
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
    num_files = 0
    degree = 0

    try:
        opts, args = getopt.getopt(
            argv,
            "h:f:n:d:",
            ["file_path=", "num_files=", "x_rez=", "y_rez="],
        )
    except getopt.GetoptError:
        print(
            "\nSyntax: main.py -f <file path to train dir> -n <num of files in dir> -d <degree of rotation>\n"
        )
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print(
                "\nSyntax: main.py -f <file path to train dir> -n <num of files in dir> -d <degree of rotation>\n"
            )
        elif opt in ("-f", "--file_path"):
            file_path = arg
        elif opt in ("-n", "--num_files"):
            num_files = int(arg)
        elif opt in ("-d", "--degree"):
            degree = int(arg)

    rotate_handler(file_path, num_files, degree)


if __name__ == "__main__":
    main(sys.argv[1:])

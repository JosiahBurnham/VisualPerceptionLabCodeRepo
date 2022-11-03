"""
File    : main.py
Author  : J.Burnham
Date    : 01/03/2022
Org     : FGCU
Purpose : Main file for Resize Sets program, splits the entire dataset to be resized into
          4 different chuncks to allow for multi threading the task for better performance.
"""

import getopt
import sys

import numpy as np
import scipy.io
from Scripts.load_data import LoadData
from Scripts.resize_sets import ResizeSets


def resize_handler(file_path, num_files, x_rez, y_rez):
    """
    Handles the threading of the task of resizeing datasets that are stored in .mat files with
    dict labels that are ["image_patches", "category_labels"]. this method uses four threads
    to optimize this task. it then joins the thread data back together, and outputs the resized
    data to another .mat file with the same dict labels

    @param file_path:   - the file path to the data dir where the orginal data is stored
    @param num_files:   - the number of files in the directory that is to be resized
    @param x_rez:       - the desired x resolution
    @param y_rez:       - the desired y resolution
    """

    x_data_sets = []
    y_data_sets = []
    iters = 0   # iteration of the for loop
    start_file = 1  # the first file in the data dir should be labeled with a one on the end
    for i in range(4):
        if iters == 0:  # if the number of files does not divide 4 add the remainder to thread 1
            data_set_files = (num_files // 4) + (num_files % 4)
        else:
            data_set_files = (num_files // 4)

        loadData = LoadData(file_path=file_path,
                            num_files=data_set_files, start_file=start_file)
        (x_data, y_data) = loadData.load_file_data()

        x_data_sets.append(x_data)
        y_data_sets.append(y_data)

        iters += 1
        start_file += data_set_files  # set the next file to load

    # define all threads
    first_data_thread = ResizeSets(x_data_sets[0], x_rez, y_rez, "Subset 1")
    second_data_thread = ResizeSets(x_data_sets[1], x_rez, y_rez, "Subset 2")
    third_data_thread = ResizeSets(x_data_sets[2], x_rez, y_rez, "Subset 3")
    fourth_data_thread = ResizeSets(x_data_sets[3], x_rez, y_rez, "Subset 4")

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
    fully_resized_set = np.concatenate((first_data_thread.resized_set, second_data_thread.resized_set,
                                       third_data_thread.resized_set, fourth_data_thread.resized_set))

    all_y_data = np.concatenate(
        (y_data_sets[0], y_data_sets[1], y_data_sets[2], y_data_sets[3]))

    # Save Resized Data set to a .mat file
    matfile_name = "./Resized_Sets/" + "Resized_Set" + \
        "_" + str(fully_resized_set.shape) + ".mat"
    save_dict = {"image_patches": fully_resized_set,
                 "category_labels": all_y_data}
    try:
        scipy.io.savemat(matfile_name, save_dict)
    except FileNotFoundError:
        f = open(matfile_name, "x")
        f.close()
        scipy.io.savemat(matfile_name, save_dict)


def main(argv):
    """
    gets the arguments passed from the command line, then passes those args to the resize method
    @param argv: The arguments from the CLI
    """
    file_path = ""
    num_files = 0
    x_rez = 0
    y_rez = 0

    try:
        opts, args = getopt.getopt(
            argv,
            "h:f:n:x:y:",
            ["file_path=", "num_files=", "x_rez=", "y_rez="],
        )
    except getopt.GetoptError:
        print(
            "\nSyntax: main.py -f <file path to train dir> -n <num of files in dir>,-x <x resolution>,-y <y resolution>\n"
        )
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print(
                "\nSyntax: main.py -f <file path to train dir> -n <num of files in dir>,-x <x resolution>,-y <y resolution>\n"
            )
        elif opt in ("-f", "--file_path"):
            file_path = arg
        elif opt in ("-n", "--num_files"):
            num_files = int(arg)
        elif opt in ("-x", "--x_rez"):
            x_rez = int(arg)
        elif opt in ("-y", "--y_rez"):
            y_rez = int(arg)

    resize_handler(file_path, num_files, x_rez, y_rez)


if __name__ == "__main__":
    main(sys.argv[1:])

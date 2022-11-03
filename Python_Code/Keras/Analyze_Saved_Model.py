"""
File:   Analyze_Model.py
Author: Josiah Burnham (1/2022)
Org:    FGCU
Desc:   Find the ACC metrics of any given trained model
"""

import sys
import getopt
import numpy as np
from tracemalloc import start
from tensorflow import keras

from Scripts.load_data import LoadData




def analyze(file_path, num_files, start_file):


    data = LoadData(file_path=file_path, num_files=num_files, start_file = start_file, num_img=68000, x_shape=40, y_shape=2)
    (x_data, y_data) = data.load_file_data()

    # Load test sets
    file_65_x = np.asarray(x_data[64000:65000])
    file_65_y = np.asarray(y_data[64000:65000])

    file_66_x = np.asarray(x_data[65000:66000])
    file_66_y = np.asarray(y_data[65000:66000])

    file_67_x = np.asarray(x_data[66000:67000])
    file_67_y = np.asarray(y_data[66000:67000])

    file_68_x = np.asarray(x_data[67000:68000])
    file_68_y = np.asarray(y_data[67000:68000])


    model = keras.models.load_model("C:\\Users\\jjburnham0705\\Desktop\\FGCU_CV_Research\\One_Stage_Gabor\\gabor_oset2_24x24")   

    file_65_y = file_65_y[:,1]
    file_66_y = file_66_y[:,1]
    file_67_y = file_67_y[:,1]
    file_68_y = file_68_y[:,1]

    avg = 0

    test_scores = model.evaluate(file_65_x, file_65_y, verbose=2)
    avg += test_scores[1]
    print("File 65 : ", test_scores[1])

    test_scores = model.evaluate(file_66_x, file_66_y, verbose=2)
    avg += test_scores[1]
    print("File 66 : ", test_scores[1])

    test_scores = model.evaluate(file_67_x, file_67_y, verbose=2)
    avg += test_scores[1]
    print("File 67 : ", test_scores[1])

    test_scores = model.evaluate(file_68_x, file_68_y, verbose=2)
    avg += test_scores[1]
    print("File 68 : ", test_scores[1])

    avg /= 4
    print("Average of four sets : ",avg)
        



def main(argv):
    """
    gets the arguments passed from the command line, then passes those args to the train model function
    @param argv: The arguments from the CLI
    """
    file_path = " "
    num_files = 0
    start_file = 0

    try:
        opts, args = getopt.getopt(argv, "h:f:n:s:",
                                   ["file_path=", "num_files=", "start_file="])
    except getopt.GetoptError:
        print("\nSyntax: Analyze_Model.py -f <file_path> -n <num_files> -s <start_file> \n")
        sys.exit(2)

    # Parse through all the arguments that were passed
    for opt, arg in opts:
        if opt == '-h':
            print("\nSyntax: Analyze_Model.py -f <file_path> -n <num_files> -s <start_file> \n")
            sys.exit()
        elif opt in ("-f", "--file_path"):
            file_path = arg
        elif opt in ("-n", "--num_files"):
            num_files = int(arg)
        elif opt in ("-s", "--start_file"):
            start_file = int(arg)

    analyze(file_path, num_files, start_file)


if __name__ == "__main__":
    main(sys.argv[1:])
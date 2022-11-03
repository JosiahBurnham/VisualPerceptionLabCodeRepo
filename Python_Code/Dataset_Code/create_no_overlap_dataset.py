import numpy as np
from tensorflow import keras
import scipy.io 
from scipy.io import savemat

from load_dataset import LoadDataset
from dataset_KFold import KFoldDataset
from prep_dataset import PrepDataset

"""
dir_path = "CV_SHAD_CSHD_BYIM\\GRY\\40x40\\Fold_"
num_files = 145

x_all = np.zeros((0,1600))
y_all = np.zeros((0,2)) 

for i in range(5): # num folds
    fold_path = dir_path + str(i+1) + "\\";
    ld = LoadDataset(fold_path)
    (x,y) = ld.load_dataset()
    
    x_all = np.append(x_all, x, axis=0)
    y_all = np.append(y_all, y, axis=0)




kfold = KFoldDataset((x_all, y_all), num_folds=5)

(x_train, y_train), (x_test, y_test) =  kfold.make_folds()

# this will be the only part that you will need if you use the new load_dataset class and the new saved .mat file
#(x_train_fold, y_train_fold), (x_test_fold, y_test_fold) = PrepDataset(x_train, y_train, x_test, y_test, 0).prep_epoch_data()


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

matDict = {
    "x_train": x_train,
    "y_train": y_train,
    "x_test" : x_test,
    "y_test" : y_test
}

savemat("SHAD_CSHD_FOLDS_5.mat", matDict)
    """

dir_path = "Data\\SHAD_EDGE_FOLDS_5"


file_data = scipy.io.loadmat(dir_path)

x_train = file_data["x_train"]
x_test = file_data["x_test"]
y_test = file_data["y_test"]
y_train = file_data["y_train"]

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

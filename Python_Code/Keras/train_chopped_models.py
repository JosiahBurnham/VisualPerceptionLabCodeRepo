from ast import Load
import numpy as np
import tensorflow as tf
from tensorflow import keras


from Scripts.load_dataset import LoadDataset
from AlexNet.AlexNet import AlexNet
from VGG16.VGG16_Model import VGG_16
from VGG19.VGG19_Model import VGG_19




def main():

    dir_path1 = "C:\\Users\\jjburnham0705\\OneDrive - Florida Gulf Coast University\\Dataset Stuff\\Circularly_Crop_Image_Sets\\Circularly_Cropped_Sets\\CV_SHAD_EDGE_BYIM\\Rotated_Left"
    dir_path2 = "C:\\Users\\jjburnham0705\\OneDrive - Florida Gulf Coast University\\Dataset Stuff\\Circularly_Crop_Image_Sets\\Circularly_Cropped_Sets\\CV_SHAD_EDGE_BYIM\\Rotated_Right"
    
    loadDB = LoadDataset(dir_path1, dir_path2, x_shape=64, y_shape=2, num_folds=5, holdout_fold=1)
    (x_data, y_data), (test_x, test_y) = loadDB.load_dataset()

    print(x_data.shape)

    print(y_data[0])
    print(y_data[12740])

    anetData = np.load("AlexNet\\AlexNet_WD\\AlexNet_WD.npy", mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
    vgg16Data = np.load("VGG16\\VGG_16_WD\\VGG_16_Weights.npy", mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
    vgg19Data = np.load("VGG19\\VGG_19_WD\\VGG_19_Weights.npy", mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
    
    vgg16 = VGG_16(vgg16Data, 2, l2= 0.001, pool_rate=4, output=True)
    vgg19 = VGG_19(vgg19Data, 2, output=False)
    anet = AlexNet(anetData, 2, output=False)

    model1 = vgg16.get_model(4)



    model1.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
              loss=keras.losses.CategoricalCrossentropy(),
              metrics=[keras.metrics.BinaryAccuracy()])
    model1.summary()

    history = model1.fit(x_data, y_data, batch_size=50, epochs=3, verbose=1)


    print("...EVALUATING MODEL ON TEST BATCH...")
    test_scores = model1.evaluate(test_x, test_y, verbose=1)


main()



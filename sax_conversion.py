"""
Created on Fri Dec 6 21:46:23 2017
@author: avrahami & Bolless
"""
# use python 3.6 for running. Most have tensorflow installed

from __future__ import print_function

from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.utils import np_utils
import numpy as np
import keras
from keras.callbacks import ReduceLROnPlateau
import pandas as pd
from sklearn.metrics import confusion_matrix
np.random.seed(813306)


def readucr(filename):
    data = np.loadtxt(filename, delimiter=',')
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y

# should be much higher, something like 5000
nb_epochs = 100
multivariate = False
# flist = ['Adiac', 'Beef', 'CBF', 'ChlorineConcentration', 'CinC_ECG_torso', 'Coffee', 'Cricket_X', 'Cricket_Y', 'Cricket_Z',
# 'DiatomSizeReduction', 'ECGFiveDays', 'FaceAll', 'FaceFour', 'FacesUCR', '50words', 'FISH', 'Gun_Point', 'Haptics',
# 'InlineSkate', 'ItalyPowerDemand', 'Lighting2', 'Lighting7', 'MALLAT', 'MedicalImages', 'MoteStrain', 'NonInvasiveFatalECG_Thorax1',
# 'NonInvasiveFatalECG_Thorax2', 'OliveOil', 'OSULeaf', 'SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII', 'StarLightCurves', 'SwedishLeaf', 'Symbols',
# 'synthetic_control', 'Trace', 'TwoLeadECG', 'Two_Patterns', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y', 'uWaveGestureLibrary_Z', 'wafer', 'WordsSynonyms', 'yoga']

data_location = 'C:\\Users\\abrahami\\Documents\\Private\\Uni\\BGU\\time_series\\project\\data'
flist = ['NonInvasiveFatalECG_Thorax1']
if not multivariate:
    fname = flist[0]
    x_train, y_train = readucr(data_location + '\\' + fname + '_TRAIN')
    x_test, y_test = readucr(data_location + '\\' + fname + '_TEST')
    nb_classes = len(np.unique(y_test))

    # normalization
    y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (nb_classes - 1)
    y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (nb_classes - 1)
    batch_size = min(x_train.shape[0] / 10, 16)

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    x_train_mean = x_train.mean()
    x_train_std = x_train.std()
    x_train = (x_train - x_train_mean) / (x_train_std)

    # x_test_min = np.min(x_test, axis = 1, keepdims=1)
    # x_test_max = np.max(x_test, axis = 1, keepdims=1)
    x_test = (x_test - x_train_mean) / (x_train_std)

    # x_train = x_train.reshape(x_train.shape + (1,))
    # x_test = x_test.reshape(x_test.shape + (1,))

    x = Input(shape=x_train.shape[1:])
    y = Dropout(0.1)(x)
    y = Dense(500, activation='relu')(x)
    y = Dropout(0.2)(y)
    y = Dense(500, activation='relu')(y)
    y = Dropout(0.2)(y)
    y = Dense(500, activation='relu')(y)
    y = Dropout(0.3)(y)
    out = Dense(nb_classes, activation='softmax')(y)

    model = Model(input=x, output=out)

    optimizer = keras.optimizers.Adadelta()
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                                  patience=200, min_lr=0.1)

    # a bit strange why is the x_test and y_test being used at all????
    hist = model.fit(x_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs,
                     verbose=1, validation_data=(x_test, Y_test),
                     # callbacks = [TestCallback((x_train, Y_train)), reduce_lr, keras.callbacks.TensorBoard(log_dir='./log'+fname, histogram_freq=1)])
                     callbacks=[reduce_lr])

    # Print the testing results which has the lowest training loss.
    log = pd.DataFrame(hist.history)
    print(log.loc[log['loss'].idxmin]['loss'], log.loc[log['loss'].idxmin]['val_acc'])

    test_proba_prediction = model.predict(x=x_test,batch_size=batch_size, verbose=1)
    test_prediction = np.apply_along_axis(func1d=np.argmax, axis=1, arr=test_proba_prediction)
    conf_matrix = confusion_matrix(y_true=y_test.astype(int), y_pred=test_prediction.astype(int))
    error_rate = 1-np.trace(conf_matrix)*1.0/conf_matrix.sum()*1.0
    print("\nError rate of current run is {}". format(error_rate))

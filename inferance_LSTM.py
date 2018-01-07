# define the LSTM model
import numpy as np

import pickle
from sklearn.metrics import confusion_matrix
# Small LSTM Network to Generate Text for Alice in Wonderland
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import pandas as pd
import sys

path = "data/"
df = pd.read_csv(path + 'uWaveGestureLibrary_X_TRAIN.txt')


df_y = pd.read_csv(path + 'uWaveGestureLibrary_X_TRAIN')
labels = df_y.values[:,0]
y = np_utils.to_categorical(labels)


df_test = pd.read_csv(path + 'uWaveGestureLibrary_X_TEST.txt')

df_y_test = pd.read_csv(path + 'uWaveGestureLibrary_X_TEST')
labels_test = df_y_test.values[:,0]





x = []
flat_x = []
for i in range(len(df.values)):
    flat_x.extend(df.values[i][0].split())
    x.append(df.values[i][0].split())
x_test = []
for i in range(len(df_test.values)):
    flat_x.extend(df_test.values[i][0].split())
    x_test.append(df_test.values[i][0].split())


chars = sorted(list(set(flat_x)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

n_vocab = len(chars)
print("Total Vocab: ", n_vocab)

dataX = [[char_to_int[char] for char in x_ ]for x_ in x]
x_onehot = []
for X in dataX:
    x_onehot.append(np.empty([len(x[0]),n_vocab]))
    for i,x_ in enumerate(X):
        x_onehot[-1][i,:] = np_utils.to_categorical(x_,num_classes=n_vocab)
onehot = np.array(x_onehot)

print(onehot[0][0])

dataX_test = [[char_to_int[char] for char in x_ ]for x_ in x_test]
x_onehot_test = []
for X in dataX_test:
    x_onehot_test.append(np.empty([len(x_test[0]),n_vocab]))
    for i,x_ in enumerate(X):
        x_onehot_test[-1][i,:] = np_utils.to_categorical(x_,num_classes=n_vocab)
onehot_test = np.array(x_onehot_test)

print(onehot[0][0])
print(onehot_test[0][0])

model = Sequential()
model.add(LSTM(256, input_shape=(len(onehot[0]),  len(onehot[0][0]))))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
filename = "weights-improvement-99-0.0001.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

prediction_train = model.predict(onehot, verbose=0)
train_prediction=[]
for pred in prediction_train:
    train_prediction.append(np.argmax(pred))

prediction_test = model.predict(onehot_test, verbose=0)
test_prediction=[]
for pred in prediction_test:
    test_prediction.append(np.argmax(pred))

print(train_prediction)
print(labels)
print(test_prediction)
print(labels_test)

with open('y_pred', 'wb') as fp:
    pickle.dump(train_prediction, fp)
conf_matrix = confusion_matrix(y_true=np.asarray(labels).astype(int), y_pred=np.asarray(train_prediction).astype(int))
error_rate = 1 - np.trace(conf_matrix) * 1.0 / conf_matrix.sum() * 1.0
print("Error rate of current run is {}".format(error_rate))
with open('y_pred_test', 'wb') as fp:
    pickle.dump(test_prediction, fp)
conf_matrix = confusion_matrix(y_true=np.asarray(labels_test).astype(int), y_pred=np.asarray(test_prediction).astype(int))
error_rate = 1 - np.trace(conf_matrix) * 1.0 / conf_matrix.sum() * 1.0
print("Error rate of current run is {}".format(error_rate))

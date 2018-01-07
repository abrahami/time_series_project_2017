
import numpy as np


import pickle
from sklearn.metrics import confusion_matrix

# Small LSTM Network to Generate Text for Alice in Wonderland
from keras.models import Sequential
from keras.layers import Dense, Merge
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import pandas as pd

def onehot(df, df_test):
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

    dataX = [[char_to_int[char] for char in x_] for x_ in x]
    x_onehot = []
    for X in dataX:
        x_onehot.append(np.empty([len(x[0]), n_vocab]))
        for i, x_ in enumerate(X):
            #         print(x_)
            x_onehot[-1][i, :] = np_utils.to_categorical(x_, num_classes=n_vocab)

    dataX_test = [[char_to_int[char] for char in x_] for x_ in x_test]
    x_onehot_test = []
    for X in dataX_test:
        x_onehot_test.append(np.empty([len(x_test[0]), n_vocab]))
        for i, x_ in enumerate(X):
            x_onehot_test[-1][i, :] = np_utils.to_categorical(x_, num_classes=n_vocab)

    return np.array(x_onehot), np.array(x_onehot_test)


def getdata(filename):
    path = "data/"
    df = pd.read_csv(path + filename + '_TRAIN.txt')
    #print(df)
    path = "data/"
    df_y = pd.read_csv(path + filename + '_TRAIN')
    labels = df_y.values[:,0]
    y = np_utils.to_categorical(labels)


    df_test = pd.read_csv(path + filename + '_TEST.txt')

    df_y_test = pd.read_csv(path + filename + '_TEST')
    labels_test = df_y_test.values[:,0]
    return df, labels, y, df_test, labels_test

x_df, labels, y, x_df_test, labels_test = getdata("NonInvasiveFatalECG_Thorax1")
y_df, _, _, y_df_test, _ = getdata("NonInvasiveFatalECG_Thorax2")

x_onehot, x_onehot_test = onehot(x_df, x_df_test)
y_onehot, y_onehot_test = onehot(y_df, y_df_test)




print(x_onehot[0][0])
print(y_onehot[0][0])

# define the LSTM model
model_a = Sequential()
model_a.add(LSTM(256, input_shape=(len(x_onehot[0]),  len(x_onehot[0][0]))))
model_a.add(Dropout(0.2))

model_b = Sequential()
model_b.add(LSTM(256, input_shape=(len(y_onehot[0]),  len(y_onehot[0][0]))))
model_b.add(Dropout(0.2))


model = Sequential()
model.add(Merge([model_a, model_b],mode='sum'))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
print(model)
# fit the model
model.fit([x_onehot, y_onehot], y, epochs=50, batch_size=64, callbacks=callbacks_list)



prediction = model.predict([x_onehot, y_onehot], verbose=0)
train_prediction=[]
for pred in prediction:
    train_prediction.append(np.argmax(pred))

prediction_test = model.predict([x_onehot_test, y_onehot_test], verbose=0)
test_prediction=[]
for pred in prediction_test:
    test_prediction.append(np.argmax(pred))


with open('y_pred', 'wb') as fp:
    pickle.dump(test_prediction, fp)
conf_matrix = confusion_matrix(y_true=np.asarray(labels).astype(int), y_pred=np.asarray(train_prediction).astype(int))
error_rate = 1 - np.trace(conf_matrix) * 1.0 / conf_matrix.sum() * 1.0
print("Error rate of current run is {}".format(error_rate))

conf_matrix = confusion_matrix(y_true=np.asarray(labels_test).astype(int), y_pred=np.asarray(test_prediction).astype(int))
error_rate = 1 - np.trace(conf_matrix) * 1.0 / conf_matrix.sum() * 1.0
print("Error rate of current run is {}".format(error_rate))


import numpy as np

import logging
import pickle
from sklearn.metrics import confusion_matrix

# Small LSTM Network to Generate Text for Alice in Wonderland
from keras.models import Sequential, Model
from keras.layers import Dense, Merge, Flatten, Concatenate
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Conv1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import *
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import Nadam
import pandas as pd

logging.basicConfig(filename='Run.log', level=logging.DEBUG)
logging.info('Started')
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
    #print("Total Vocab: ", n_vocab)

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

def readucr(filename):
    data = np.loadtxt(filename, delimiter=',')
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y


def getdata(filename):
    path = "data/"
    df = pd.read_csv(path + filename + '_TRAIN.txt', header=None)
    logging.info('the SAX train Len: ' + str(len(df)))
    path = "data/"
    x_train, y_train = readucr(path + filename + '_TRAIN')
    logging.info('the train Len: ' + str(len(x_train)))
    #df_orign = pd.read_csv(path + filename + '_TRAIN')
    #labels = df_orign.values[:,0]
    labels = y_train
    y = np_utils.to_categorical(labels)
    logging.info('the Y Len: '+str(len(y)))


    df_test = pd.read_csv(path + filename + '_TEST.txt', header=None)
    logging.info('the SAX test Len: ' + str(len(df_test)))
    x_test, labels_test = readucr(path + filename + '_TEST')
    logging.info('the test Len: ' + str(len(x_test)))
    logging.info('the Y test Len: ' + str(len(labels_test)))
    #df_orign_test = pd.read_csv(path + filename + '_TEST')
    #labels_test = df_orign_test.values[:,0]
    x_train_mean = x_train.mean()
    x_train_std = x_train.std()
    x_train = (x_train - x_train_mean) / (x_train_std)
    x_test = (x_test - x_train_mean) / (x_train_std)
    return df, labels, y, df_test, labels_test, x_train, x_test



x_train_combined = list()
x_test_combined = list()
x_df, labels, y, x_df_test, labels_test, x_train, x_test = getdata("NonInvasiveFatalECG_Thorax1")
x_train_combined.append(x_train)
x_test_combined.append(x_test)
y_df, _, _, y_df_test, _, x_train, x_test = getdata("NonInvasiveFatalECG_Thorax2")
x_train_combined.append(x_train)
x_test_combined.append(x_test)

x_train_array = np.dstack(tuple(x_train_combined))
x_test_array = np.dstack(tuple(x_test_combined))

categories_dim = y.shape[1]
# window in this case is the # of features used for each instance
window = len(x_train_array[0])
# emb_size in this case is the # of dimensions (length of flist)
emb_size = len(x_train_array[0][0])

x_onehot, x_onehot_test = onehot(x_df, x_df_test)
y_onehot, y_onehot_test = onehot(y_df, y_df_test)




#print(x_onehot[0][0])
#print(y_onehot[0][0])

#abraham model building
model_ai = Sequential()
model_ai.add(Conv1D(input_shape=(window, emb_size), filters=16, kernel_size=4, padding='same'))
model_ai.add(BatchNormalization())
model_ai.add(LeakyReLU())
model_ai.add(Dropout(0.5))

model_ai.add(Conv1D(filters=8, kernel_size=4, padding='same'))
model_ai.add(BatchNormalization())
model_ai.add(LeakyReLU())
model_ai.add(Dropout(0.5))

model_ai.add(Flatten())

model_ai.add(Dense(64))
model_ai.add(BatchNormalization())
model_ai.add(LeakyReLU())

# define the LSTM model
##LSTM on SAX


model_a = Sequential()
model_a.add(LSTM(256, input_shape=(len(x_onehot[0]),  len(x_onehot[0][0]))))
model_a.add(Dropout(0.2))


model_a.add(Dense(64))
model_a.add(BatchNormalization())
model_a.add(LeakyReLU())

model_b = Sequential()
model_b.add(LSTM(256, input_shape=(len(y_onehot[0]),  len(y_onehot[0][0]))))
model_b.add(Dropout(0.2))


model_b.add(Dense(64))
model_b.add(BatchNormalization())
model_b.add(LeakyReLU())


model = Sequential()
#model.add(Merge([model_a, model_b, model_ai], mode='sum'))
model.add(Merge([model_a, model_b, model_ai], mode='concat'))
model.add(Dense(y.shape[1], activation='softmax'))
opt = Nadam(lr=0.002)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])#'adam', metrics=['accuracy'])
# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
#reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=30, min_lr=0.000001, verbose=1)
#checkpointer = ModelCheckpoint(filepath="lolkek.hdf5", verbose=1, save_best_only=True)
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
logging.info(model)
# fit the model
model.fit([x_onehot, y_onehot, x_train_array], y, epochs=150, batch_size=64, callbacks=callbacks_list,
          validation_data=([x_onehot, y_onehot, x_train_array], y))



prediction = model.predict([x_onehot, y_onehot, x_train_array], verbose=0)
train_prediction=[]
for pred in prediction:
    train_prediction.append(np.argmax(pred))

prediction_test = model.predict([x_onehot_test, y_onehot_test, x_test_array], verbose=0)
test_prediction=[]
for pred in prediction_test:
    test_prediction.append(np.argmax(pred))


with open('y_pred', 'wb') as fp:
    pickle.dump(test_prediction, fp)
conf_matrix = confusion_matrix(y_true=np.asarray(labels).astype(int), y_pred=np.asarray(train_prediction).astype(int))
error_rate = 1 - np.trace(conf_matrix) * 1.0 / conf_matrix.sum() * 1.0
logging.info("Error rate of current run is {}".format(error_rate))

conf_matrix = confusion_matrix(y_true=np.asarray(labels_test).astype(int), y_pred=np.asarray(test_prediction).astype(int))
error_rate = 1 - np.trace(conf_matrix) * 1.0 / conf_matrix.sum() * 1.0
logging.info("Error rate of current run is {}".format(error_rate))
logging.info('Finished')

# source of code: https://github.com/Rachnog/Deep-Trading/tree/master/multivariate
import pandas as pd
import numpy as np
from keras.utils import np_utils
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import *
from keras.optimizers import Nadam
from keras.initializers import *
from sklearn.metrics import confusion_matrix
import sax_word as pysax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# general functions created in Rachnog's git
def shuffle_in_unison(a, b):
    # courtsey http://stackoverflow.com/users/190280/josh-bleecher-snyder
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def create_Xt_Yt(X, y, percentage=0.7):
    p = int(len(X) * percentage)
    X_train = X[0:p]
    Y_train = y[0:p]

    X_train, Y_train = shuffle_in_unison(X_train, Y_train)

    X_test = X[p:]
    Y_test = y[p:]

    return X_train, X_test, Y_train, Y_test


def remove_nan_examples(data):
    newX = []
    for i in range(len(data)):
        if np.isnan(data[i]).any() == False:
            newX.append(data[i])
    return newX


def readucr(filename):
    data = np.loadtxt(filename, delimiter=',')
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y

def sax_data_prep(x_train, x_test, sax_obj=pysax.SAXModel(window=20, stride=5, nbins=5, alphabet="ABCD")):
    tfidf_obj = TfidfVectorizer(norm='l2', min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True)
    svd_obj = TruncatedSVD(n_components=1)
    train_sax_representation = []

    for x in x_train:
        symbols = sax_obj.symbolize_signal(x)
        train_sax_representation.append(' '.join(symbols))
    test_sax_representation = []
    for x in x_test:
        symbols = sax_obj.symbolize_signal(x)
        test_sax_representation.append(' '.join(symbols))

    train_tfidf = tfidf_obj.fit_transform(train_sax_representation)
    train_svd = svd_obj.fit_transform(train_tfidf)
    test_tfidf = tfidf_obj.transform(test_sax_representation)
    test_svd = svd_obj.transform(test_tfidf)
    # converting the single svd value to the format we will use in the NN
    train_svd_duplicated_list = [train_svd for x in range(x_train.shape[1])]
    train_svd_duplicated_array = np.array(train_svd_duplicated_list).squeeze().transpose()
    test_svd_duplicated_list = [test_svd for x in range(x_test.shape[1])]
    test_svd_duplicated_array = np.array(test_svd_duplicated_list).squeeze().transpose()
    return train_svd_duplicated_array, test_svd_duplicated_array


def run_algo(dataset, ephocs=100, with_sax=False):
    if dataset == 'trading_data':
        data_original = pd.read_csv(data_location + "\\multivariate_example_data.csv")[::-1]
        openp = data_original.ix[:, 'Open'].tolist()
        highp = data_original.ix[:, 'High'].tolist()
        lowp = data_original.ix[:, 'Low'].tolist()
        closep = data_original.ix[:, 'Adj Close'].tolist()
        volumep = data_original.ix[:, 'Volume'].tolist()

        window = 30
        emb_size = 5
        step = 1
        forecast = 1

        X, Y = [], []
        for i in range(0, len(data_original), step):
            try:
                o = openp[i:i+window]
                h = highp[i:i+window]
                l = lowp[i:i+window]
                c = closep[i:i+window]
                v = volumep[i:i+window]

                o = (np.array(o) - np.mean(o)) / np.std(o)
                h = (np.array(h) - np.mean(h)) / np.std(h)
                l = (np.array(l) - np.mean(l)) / np.std(l)
                c = (np.array(c) - np.mean(c)) / np.std(c)
                v = (np.array(v) - np.mean(v)) / np.std(v)

                x_i = closep[i:i+window]
                y_i = closep[i+window+forecast]

                last_close = x_i[-1]
                next_close = y_i

                if last_close < next_close:
                    y_i = [1, 0]
                else:
                    y_i = [0, 1]

                x_i = np.column_stack((o, h, l, c, v))

            except Exception as e:
                break

            X.append(x_i)
            Y.append(y_i)

        X, Y = np.array(X), np.array(Y)
        x_train, x_test, y_train_array, y_test_array = create_Xt_Yt(X, Y)

        x_train_array = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], emb_size))
        x_test_array = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], emb_size))
        categories_dim = len(y_train_array[0])

    elif dataset == "ECG":
        flist = ['NonInvasiveFatalECG_Thorax1', 'NonInvasiveFatalECG_Thorax2']
        x_train_combined = list()
        x_test_combined = list()
        y_train_combined = list()
        y_test_combined = list()
        for idx, fname in enumerate(flist):
            x_train, y_train = readucr(data_location + '\\' + fname + '_TRAIN')
            x_test, y_test = readucr(data_location + '\\' + fname + '_TEST')
            nb_classes = len(np.unique(y_test))

            '''
            # under-sampling, only for debugging purposes!!!!!
            x_train = x_train[0:300]
            y_train = y_train[0:300]
            x_test = x_test[0:200]
            y_test = y_test[0:200]
            nb_classes = len(np.unique(y_test))
            '''

            # normalization
            y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (nb_classes - 1)
            y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (nb_classes - 1)
            x_train_mean = x_train.mean()
            x_train_std = x_train.std()
            x_train = (x_train - x_train_mean) / (x_train_std)
            x_test = (x_test - x_train_mean) / (x_train_std)

            x_train_combined.append(x_train)
            y_train_combined.append(y_train)
            x_test_combined.append(x_test)
            y_test_combined.append(y_test)
            # SAX words usage (Bolless code in some way) - if needed
            if with_sax:
                sax_obj = pysax.SAXModel(window=20, stride=5, nbins=5, alphabet="ABCD")
                train_svd, test_svd = sax_data_prep(sax_obj=sax_obj, x_train=x_train, x_test=x_test)
                # need to add elements to the exiting lists of data-frames
                x_train_combined.append(train_svd)
                x_test_combined.append(test_svd)

        x_train_array = np.dstack(tuple(x_train_combined))
        x_test_array = np.dstack(tuple(x_test_combined))
        y_train_array = to_categorical(y_train_combined[0])
        y_test_array = to_categorical(y_test_combined[0])

        categories_dim = len(y_train_array[0])
        # window in this case is the # of features used for each instance
        window = len(x_train_array[0])
        # emb_size in this case is the # of dimensions (length of flist)
        emb_size = len(x_train_array[0][0])

    elif dataset == "ECG":
        flist = ['NonInvasiveFatalECG_Thorax1', 'NonInvasiveFatalECG_Thorax2']
        x_train_combined = list()
        x_test_combined = list()
        y_train_combined = list()
        y_test_combined = list()
        for idx, fname in enumerate(flist):
            x_train, y_train = readucr(data_location + '\\' + fname + '_TRAIN')
            x_test, y_test = readucr(data_location + '\\' + fname + '_TEST')
            nb_classes = len(np.unique(y_test))

            # under-sampling, only for debugging purposes!!!!!
            x_train = x_train[0:300]
            y_train = y_train[0:300]
            x_test = x_test[0:200]
            y_test = y_test[0:200]
            nb_classes = len(np.unique(y_test))

            # normalization
            y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (nb_classes - 1)
            y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (nb_classes - 1)
            x_train_mean = x_train.mean()
            x_train_std = x_train.std()
            x_train = (x_train - x_train_mean) / (x_train_std)
            x_test = (x_test - x_train_mean) / (x_train_std)

            x_train_combined.append(x_train)
            y_train_combined.append(y_train)
            x_test_combined.append(x_test)
            y_test_combined.append(y_test)
            # SAX words usage (Bolless code in some way) - if needed
            if with_sax:
                sax_obj = pysax.SAXModel(window=20, stride=5, nbins=5, alphabet="ABCD")
                train_svd, test_svd = sax_data_prep(sax_obj=sax_obj, x_train=x_train, x_test=x_test)
                # need to add elements to the exiting lists of data-frames
                x_train_combined.append(train_svd)
                x_test_combined.append(test_svd)

        x_train_array = np.dstack(tuple(x_train_combined))
        x_test_array = np.dstack(tuple(x_test_combined))
        y_train_array = to_categorical(y_train_combined[0])
        y_test_array = to_categorical(y_test_combined[0])

        categories_dim = len(y_train_array[0])
        # window in this case is the # of features used for each instance
        window = len(x_train_array[0])
        # emb_size in this case is the # of dimensions (length of flist)
        emb_size = len(x_train_array[0][0])

    else:
        print("such data is not supported. Try either 'ECG' or 'trading_data'")
        return -1
    # model building
    model = Sequential()
    model.add(Conv1D(input_shape=(window, emb_size), filters=16, kernel_size=4, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))

    model.add(Conv1D(filters=8, kernel_size=4, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Dense(categories_dim))
    model.add(Activation('softmax'))

    opt = Nadam(lr=0.002)

    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=30, min_lr=0.000001, verbose=1)
    checkpointer = ModelCheckpoint(filepath="lolkek.hdf5", verbose=1, save_best_only=True)

    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train_array, y_train_array, nb_epoch=ephocs, batch_size=128, verbose=1,
                        validation_data=(x_test_array, y_test_array),
                        callbacks=[reduce_lr, checkpointer], shuffle=True)

    # Print the testing results which has the lowest training loss.
    log = pd.DataFrame(history.history)
    print(log.loc[log['loss'].idxmin]['loss'], log.loc[log['loss'].idxmin]['val_acc'])

    # train prediction
    train_proba_prediction = model.predict(x=x_train_array, batch_size=128, verbose=1)
    train_prediction = np.apply_along_axis(func1d=np.argmax, axis=1, arr=train_proba_prediction)
    y_train_flatten = np.array([idx for i in y_train_array for idx, j in enumerate(list(i)) if j])
    conf_matrix = confusion_matrix(y_true=y_train_flatten.astype(int), y_pred=train_prediction.astype(int))
    error_rate = 1 - np.trace(conf_matrix) * 1.0 / conf_matrix.sum() * 1.0
    print("\nError rate of current run over train data is {}".format(error_rate))

    test_proba_prediction = model.predict(x=x_test_array, batch_size=128, verbose=1)
    test_prediction = np.apply_along_axis(func1d=np.argmax, axis=1, arr=test_proba_prediction)
    y_test_flatten = np.array([idx for i in y_test_array for idx, j in enumerate(list(i)) if j])
    conf_matrix = confusion_matrix(y_true=y_test_flatten.astype(int), y_pred=test_prediction.astype(int))
    error_rate = 1 - np.trace(conf_matrix) * 1.0 / conf_matrix.sum() * 1.0
    print("\nError rate of current run over test data is {}".format(error_rate))

if __name__ == '__main__':
    # should be one out of the two
    dataset = "ECG"#"trading_data"
    ephocs = 100
    data_location = "C:\\Users\\abrahami\\Documents\\Private\\Uni\\BGU\\time_series\\project\\data"
    run_algo(dataset=dataset, ephocs=ephocs, with_sax=False)

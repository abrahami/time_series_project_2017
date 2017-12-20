import matplotlib.pyplot as plt

from collections import Counter
import sax_word as pysax
import numpy as np

#in Scikit-Learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import pickle


#reload(pysax)
"""
sax = pysax.SAXModel(window=3, stride=2)
print(sax.sym2vec)

## test normalization
sax = pysax.SAXModel(window=3, stride=2)
list(sax.sliding_window_index(10))
ws = np.random.random(10)
print(ws.mean(), ws.std())
ss = sax.whiten(ws)
print(ss.mean(), ss.std())

## explore binning

from fractions import Fraction


def binpack(xs, nbins):
    xs = np.asarray(xs)
    binsize = Fraction(len(xs), nbins)
    wts = [1 for _ in range(int(binsize))] + [binsize - int(binsize)]
    pos = 0
    while pos < len(xs):
        if wts[-1] == 0:
            n = len(wts) - 1
        else:
            n = len(wts)
        yield list(zip(xs[pos:(pos + n)], wts[:n]))#add list
        pos += len(wts) - 1
        rest_wts = binsize - (1 - wts[-1])
        wts = [1 - wts[-1]] + [1 for _ in range(int(rest_wts))] + [rest_wts - int(rest_wts)]


xs = range(0, 16)
print(list(binpack(xs, 5)))
xs = range(0, 16)
print(list(binpack(xs, 4)))
xs = range(0, 5)
print(list(binpack(xs, 3)))


## test binning
sax = pysax.SAXModel(nbins=3)
print(list(sax.binpack(np.ones(5))))
print()
print(list(sax.binpack(np.ones(9))))


## explore symbolization
import pandas as pd
cutpoints = [-np.inf, -0.43, 0.43, np.inf]
xs = np.random.random(10)
v = pd.cut(xs, bins = cutpoints, labels=["A", "B", "C"])
print(v)

xs = np.random.randn(10)
print(xs)
sax = pysax.SAXModel(window=3, stride=2)
print(sax.symbolize(xs))


sax = pysax.SAXModel(nbins = 5, alphabet="ABCD")
xs = np.random.randn(20) * 2 + 1.
print(xs)
print(sax.symbolize_window(xs))


sax = pysax.SAXModel(window=20, stride = 5, nbins = 5, alphabet="ABCD")
xs = np.random.randn(103) * 2 + np.arange(103) * 0.03
plt.plot(xs)
print(sax.symbolize_signal(xs))
#plt.show()



sax = pysax.SAXModel(window=20, stride = 20, nbins = 5, alphabet="ABCD")
xs = np.random.randn(103) * 2 + np.arange(103) * 0.03
words = sax.symbolize_signal(xs)
ts_indices = sax.convert_index(word_indices=range(len(words)))
word_indices = sax.convert_index(ts_indices = range(len(xs)))
print(words)
print(ts_indices)
print(word_indices)


    #print(np.all(psymbols==symbols))


   sax = pysax.SAXModel(window=20, stride = 5, nbins = 5, alphabet="ABCD")
    xs = np.random.randn(1000000) * 2 + np.arange(1000000) * 0.03
    #plt.plot(xs)
    psymbols = sax.symbolize_signal(xs, parallel="joblib", n_jobs=30)
"""


def readucr(filename):
    data = np.loadtxt(filename, delimiter=',')
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y

'''
sax = pysax.SAXModel(window=20, stride=5, nbins=5, alphabet="ABCD")
xs = np.random.randn(1000) * 2 + np.arange(1000) * 0.03
# plt.plot(xs)
#psymbols = sax.symbolize_signal(xs, parallel="joblib", n_jobs=2)
symbols = sax.symbolize_signal(xs)
print(symbols)
print(Counter(symbols))

xs = np.random.randn(1000) * 2 + np.arange(1000) * 0.03
psymbols = sax.symbolize_signal(xs)
'''


def display_scores(vectorizer, tfidf_result):
    # http://stackoverflow.com/questions/16078015/
    scores = zip(vectorizer.get_feature_names(),
                 np.asarray(tfidf_result.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    for item in sorted_scores:
        print("{0:50} Score: {1}".format(item[0], item[1]))


data_location = 'D:\\Uni\\time analysis\\project\\NonInvasiveFatalECG_Thorax1'
flist = ['NonInvasiveFatalECG_Thorax1']
sax = pysax.SAXModel(window=20, stride=5, nbins=5, alphabet="ABCD")
for each in flist:
    fname = each
    x_train, y_train = readucr(data_location + '\\' + fname + '_TRAIN')
    x_test, y_test = readucr(data_location + '\\' + fname + '_TEST')
    nb_classes = len(np.unique(y_test))

    all_documents = [[] for _ in range(nb_classes)]
    i = 0
    for x in x_train:
        symbols = sax.symbolize_signal(x)
        #print(' '.join(symbols))
        all_documents[y_train[i].astype(int)-1].append(' '.join(symbols))
        i += 1
    documents = []
    for doc in all_documents:
        print(doc)
        documents.append(' '.join(doc))
    with open('train', 'wb') as fp:
        pickle.dump(documents, fp)

    with open('train', 'rb') as fp:
        documents = pickle.load(fp)
    all_documents_test = []
    for x in x_test:
        symbols = sax.symbolize_signal(x)
        # print(' '.join(symbols))
        all_documents_test.append(' '.join(symbols))
    with open('test', 'wb') as fp:
        pickle.dump(all_documents_test, fp)

    with open('test', 'rb') as fp:
        all_documents_test = pickle.load(fp)



    sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True)#, tokenizer=tokenize)
    sklearn_representation = sklearn_tfidf.fit_transform(documents)
    display_scores(sklearn_tfidf,sklearn_representation)

    cosine_similarities = linear_kernel(sklearn_representation[0:1], sklearn_representation).flatten()
    print(cosine_similarities)
    print(sklearn_representation[0:1])

    ''' deal with new document '''
    response = sklearn_tfidf.transform(all_documents_test)
    cosine_similarities = linear_kernel(response[0], sklearn_representation).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-1]
    print(related_docs_indices)
'''

    #print(sklearn_representation)
#print(sax.symbol_distance(symbols, psymbols))

'''



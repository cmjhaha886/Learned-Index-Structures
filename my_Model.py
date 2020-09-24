import rbf_Model_BF
import time
import numpy as np
from sklearn import svm, datasets
import sys
import datetime
import os
from sklearn.metrics import classification_report, confusion_matrix
import bloom
from bloom_filter import BloomFilter
from sklearn.model_selection import train_test_split


if __name__ == "__main__":


    np.random.seed(int(sys.argv[3]))
    X, Y = rbf_Model_BF.completely_non_linearly_separable_data(int(sys.argv[1]), float(sys.argv[2]))

    start = datetime.datetime.now() 

    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20)
    fpr_b = 0.01
    h = .02  # step size in the mesh

    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    C = 0.2  # SVM regularization parameter

    rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, Y)
    print('RBF: ')
    start = datetime.datetime.now()
    y_pred = rbf_svc.predict(X)
    end = datetime.datetime.now()
    print('========== Learned Bloom filter result =============')
    print("Learned Bloom average predict time: ", (end - start))
    y_label = [int(i) for i in Y]
    conf_matrix = confusion_matrix(y_label, y_pred)
    print(conf_matrix)
    # print(classification_report(y_label, y_pred))

    print('========== Traditional Bloom filter result =========')
    bloom = bloom.BloomFilter(len(X), fpr_b)
    for i in range(len(X)):
        if Y[i] == 1:
            bloom.add(X[i][0])
    result = []
    start = datetime.datetime.now()
    y_bloom = [bloom.check(x[0]) for x in X]
    end = datetime.datetime.now()
    print(bloom.size)
    print("Traditional Bloom average predict time: ", (end-start))
    print(confusion_matrix(y_label, y_bloom))



    
    

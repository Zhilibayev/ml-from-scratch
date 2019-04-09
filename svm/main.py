from utils import load_mat_file, Plotter, load_vocabulary, process_email, email_feature
from sklearn.svm import SVC
from svm import SVM
import numpy as np

def linear_kernel():
    X, y = load_mat_file('ex6data1.mat')
    plot = Plotter(X, y, 'on')
    plot.plot_data()

    C = 1e5

    svm = SVM(C=C, kernel='linear')
    svm.train(X, y)

    W = svm.w
    b = svm.b
    plot.visualize_boundary_linear(W, b)


def rbf_kernel():
    X, y = load_mat_file('ex6data2.mat')
    plot = Plotter(X, y, 'on')
    plot.plot_data()

    C = 1.0
    rbf_svm = SVC(C=C, kernel='rbf')

    rbf_svm.fit(X, y)

    rbf_svm.predict()


def spam_classification():
    vocabs = load_vocabulary('vocab.txt')
    X, y = load_mat_file('spamTrain.mat')
    y = y.reshape((-1, 1))
    y = y.astype(np.double)
    y[y == 0] = -1
    C = 0.1
    svm = SVM(C=C, kernel='linear', is_saved=True)
    svm.train(X, y)

    pred_train = svm.predict(X)
    print("Training accuracy:", len(pred_train[pred_train == y])/len(pred_train))

    X_test, y_test = load_mat_file('spamTest.mat')
    y_test = y_test.reshape((-1, 1))
    y_test = y_test.astype(np.double)
    y_test[y_test == 0] = -1

    pred_test = svm.predict(X_test)
    print("Testing accuracy:", len(pred_test[pred_test == y_test]) / len(pred_test))

    # Try with emailSample1.txt
    with open('emailSample1.txt') as f:
        sample_1 = f.read()
    word_indices = process_email(sample_1, vocabs)
    x = email_feature(word_indices, vocabs)
    x = x.reshape((-1, 1)).T
    is_spam = svm.predict(x)
    print("Spam" if is_spam[0] == 1 else "No spam")

spam_classification()
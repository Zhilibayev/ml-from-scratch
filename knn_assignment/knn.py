"""
Author: Giang Tran.
"""

import numpy as np
import pandas as pd
from scipy.linalg import norm
from sklearn.neighbors import KNeighborsClassifier


class KNN:
    """
    Characteristics of KNN algorithm:
        - Non-parametric ML algorithm. It means KNN doesn't have training phase to find optimal parameter.
        - Therefore, training phase and predicting phase in 1 phase make the algorithm very slow as the data larger
            and larger, also the dimension of vector is big.
        - ...

    Idea of KNN algorithm:
        - Given a specific dataset, for each row is a n-D dimension vector and the label.
        - Pre-processing dataset (normalization, standardization) into same scale (optional).
        - For any new point in predicting phase, the algorithm finds the distance between that point and all other
            points in training set (L_1, L_2, L_inf).
        - Base on K hyper-parameter, the algorithm will find K nearest neighbor and classify that point into
            which class.

    """
    _metrics = {'euclidean': '_l2_distance', 'manhattan': '_l1_distance', 'cosine': '_cosine_similarity'}

    def __init__(self, K, X, y, metric='euclidean'):
        self.K = K
        assert type(X) is np.ndarray, "X must be a numpy array"
        assert type(y) is np.ndarray, "y must be a numpy array"
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        self.metric = metric

    def _l1_distance(self, X_new):
        """
        l1 = abs(x_1 - x_2) + abs(y_1 - y_2)
        :param X_new:
        :return: ndarray manhattan distance of X_new versus all other points X.
        """
        return np.sum(np.abs(self.X - X_new), axis=1)

    def _l2_distance(self, X_new):
        """
        l2 = sqrt((x_1 - x_2)**2 + (y_1 - y_2)**2)
        :param X_new:
        :return: ndarray euclidean distance of X_new versus all other points X.
        """
        return np.sqrt(np.sum((self.X - X_new)**2, axis=1))

    def _cosine_similarity(self, X_new):
        """
        similarity = cos(alpha) = dot(A, B) / (len(A)*len(B))
        :param X_new:
        :return: ndarray cosine similarity of X_new versus all other points X.
        """
        return np.dot(self.X, X_new.T) / (norm(self.X, 2) * norm(X_new, 2))

    def predict(self, X_new):
        assert type(X_new) is np.ndarray, "Use numpy array instead."
        assert X_new.shape[1] == self.X.shape[1], "Mismatch shape."
        if self.metric not in self._metrics.keys():
            self.metric = 'euclidean'
        func = getattr(self, self._metrics[self.metric])
        dist = func(X_new)
        dist = np.argsort(dist)

        k_nearest = dist[:self.K]

        labels = self.y[k_nearest]
        max_, choose = 0, 0
        for c in self.classes:
            num = len(labels[labels == c])
            if num > max_:
                max_ = num
                choose = c
        return choose


if __name__ == '__main__':
    k = 5
    df = pd.read_csv("./data/train.csv")
    X = df.loc[:, :].values
    y = pd.read_csv("./data/trainDirection.csv").iloc[:, 0].values

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    knn = KNN(k, X, y, metric='manhattan')

    df_test = pd.read_csv("./data/testing.csv")
    X_test = df_test.drop('Direction', axis=1).iloc[:, 1:].values
    y_test = df_test.loc[:, 'Direction'].values

    print("X test shape:", X_test.shape)
    print("y test shape:", y_test.shape)

    y_pred = []
    for i in range(X_test.shape[0]):
        pred = knn.predict(X_test[i].reshape((1, 2)))
        y_pred.append(pred)

    y_pred = np.asarray(y_pred)

    print("My KNN accuracy:", len(y_test[y_pred == y_test]) / len(y_test))

    sk_knn = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
    sk_knn.fit(X, y)

    y_sk = sk_knn.predict(X_test)

    print("Sk-learn KNN accuracy:", len(y_test[y_sk == y_test]) / len(y_test))


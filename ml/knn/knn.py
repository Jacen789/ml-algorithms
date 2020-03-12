import numpy as np
from scipy import stats


class KNN(object):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        X = np.asanyarray(X)
        self._fit_X = X
        y = np.asanyarray(y)
        if y.ndim == 1 or y.ndim == 2 and y.shape[1] == 1:
            self.outputs_2d_ = False
            y = y.reshape((-1, 1))
        else:
            self.outputs_2d_ = True

        self.classes_ = []
        self._y = np.empty(y.shape, dtype=np.int)
        for k in range(self._y.shape[1]):
            classes, self._y[:, k] = np.unique(y[:, k], return_inverse=True)
            self.classes_.append(classes)

        if not self.outputs_2d_:
            self.classes_ = self.classes_[0]
            self._y = self._y.ravel()

        return self

    def predict(self, X):
        X = np.asanyarray(X)
        neigh_dist, neigh_ind = self.kneighbors(X)
        classes_ = self.classes_
        _y = self._y
        if not self.outputs_2d_:
            _y = self._y.reshape((-1, 1))
            classes_ = [self.classes_]

        n_outputs = len(classes_)
        n_samples = X.shape[0]

        y_pred = np.empty((n_samples, n_outputs), dtype=classes_[0].dtype)
        for k, classes_k in enumerate(classes_):
            mode, _ = stats.mode(_y[neigh_ind, k], axis=1)

            mode = np.asarray(mode.ravel(), dtype=np.intp)
            y_pred[:, k] = classes_k.take(mode)

        if not self.outputs_2d_:
            y_pred = y_pred.ravel()

        return y_pred

    def kneighbors(self, X=None, return_distance=True):
        X = np.asanyarray(X)
        dist = self.distances(X, self._fit_X)
        sample_range = np.arange(dist.shape[0])[:, None]
        neigh_ind = np.argpartition(dist, self.n_neighbors - 1, axis=1)
        neigh_ind = neigh_ind[:, :self.n_neighbors]
        neigh_ind = neigh_ind[sample_range,
                              np.argsort(dist[sample_range, neigh_ind])]
        if return_distance:
            result = np.vstack(dist), np.vstack(neigh_ind)
        else:
            result = np.vstack(result)

        return result

    @staticmethod
    def distances(X, Y):
        XX = (X * X).sum(axis=1)[:, np.newaxis]
        YY = (Y * Y).sum(axis=1)[np.newaxis, :]
        distances = - 2 * np.dot(X, Y.T)
        distances += XX
        distances += YY
        distances = np.maximum(distances, 0)
        return np.sqrt(distances)

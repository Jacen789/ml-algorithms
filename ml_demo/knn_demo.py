# -*- coding: utf-8 -*-

import os
import random

from ml.knn.knn import KNN


def main():
    # X = [[1, 1], [-1, 1], [-1, -1], [1, -1],
    #      [1, 2], [-1, 2], [-1, -2], [1, -2]]
    # y = [['第一象限', '符号相同'], ['第二象限', '符号不同'], ['第三象限', '符号相同'], ['第四象限', '符号不同'],
    #      ['第一象限', '符号相同'], ['第二象限', '符号不同'], ['第三象限', '符号相同'], ['第四象限', '符号不同']]
    X = [[1, 1], [-1, 1], [-1, -1], [1, -1],
         [1, 2], [-1, -2], [1, -2]]
    y = [['第一象限', '0'], ['第二象限', '0'], ['第三象限', '0'], ['第四象限', '0'],
         ['第一象限', '1'], ['第三象限', '1'], ['第四象限', '1']]
    test_X = [[1.1, 2.2], [2.2, -3.3], [-3.3, 4.4], [-4.4, -5.5]]
    neigh = KNN(n_neighbors=3)
    neigh.fit(X, y)
    rlt = neigh.predict(test_X)
    print(rlt)


if __name__ == "__main__":
    main()

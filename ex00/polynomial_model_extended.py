#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

import numpy as np


def check_matrix(matrix):
    if isinstance(matrix, np.ndarray) and matrix.size != 0 and len(matrix.shape) == 2:
        return True
    exit("Error matrix")


def add_polynomial_features(x, power):
    """Add polynomial features to matrix x by raising its columns to every power in the range
    of 1 up to the power given in argument.
    Args:
    x: has to be an numpy.array, where x.shape = (m,n) i.e. a matrix of shape m * n.
    power: has to be a positive integer, the power up to which the columns of matrix x
    are going to be raised.
    Return:
    - The matrix of polynomial features as a numpy.array, of shape m * (np),
    containing the polynomial feature values for all training examples.
    - None if x is an empty numpy.array.
    - None if x or power is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    if check_matrix(x) and isinstance(power, int):
        res = x
        for i in range(2, power + 1):
            res = np.hstack((res, x**i))
        return res


if __name__ == "__main__":
    x = np.arange(1, 11).reshape(5, 2)

    arr = add_polynomial_features(x, 3)
    print(arr)
    # print(
    #     arr
    #     == np.array(
    #         [
    #             [1, 2, 1, 4, 1, 8],
    #             [3, 4, 9, 16, 27, 64],
    #             [5, 6, 25, 36, 125, 216],
    #             [7, 8, 49, 64, 343, 512],
    #             [9, 10, 81, 100, 729, 1000],
    #         ]
    #     ),
    # )

    print(add_polynomial_features(x, 5))

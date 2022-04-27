#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

import numpy as np


def check_vector(vector):
    if (
        isinstance(vector, np.ndarray)
        and vector.size != 0
        and len(vector.shape) == 2
        and vector.shape[1] == 1
    ):
        return True
    exit("Error vector")


def iterative_l2(theta):
    """Computes the L2 regularization of a non-empty numpy.array, with a for-loop.
    Args:
    theta: has to be a numpy.array, a vector of shape n' * 1.
    Return:
    The L2 regularization as a float.
    None if theta in an empty numpy.array.
    None if theta is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    if check_vector(theta):
        ret = 0
        for i in range(1, theta.shape[0]):
            ret += theta[i][0] ** 2
        return ret


def l2(theta):
    """Computes the L2 regularization of a non-empty numpy.array, without any for-loop.
    Args:
    theta: has to be a numpy.array, a vector of shape n' * 1.
    Return:
    The L2 regularization as a float.
    None if theta in an empty numpy.array.
    None if theta is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    if check_vector(theta):
        ret = theta[1:].T @ theta[1:]
        return ret[0][0]


if __name__ == "__main__":
    x = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
    print(iterative_l2(x))
    # Output: 911.0

    print(l2(x))
    # Output: 911.0

    y = np.array([[3], [0.5], [-6]])
    print(iterative_l2(y))
    # Output: 36.25

    print(l2(y))
    # Output: 36.25

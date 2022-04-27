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


def reg_loss_(y, y_hat, theta, lambda_):
    """Computes the regularized loss of a linear regression model from two non-empty numpy.array,
    without any for loop. The two arrays must have the same shapes.
    Args:
    y: has to be an numpy.array, a vector of shape m * 1.
    y_hat: has to be an numpy.array, a vector of shape m * 1.
    theta: has to be a numpy.array, a vector of shape n * 1.
    lambda_: has to be a float.
    Return:
    The regularized loss as a float.
    None if y, y_hat, or theta are empty numpy.array.
    None if y and y_hat do not share the same shapes.
    None if y or y_hat is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    if (
        check_vector(y)
        and check_vector(y_hat)
        and y.shape == y_hat.shape
        and check_vector(theta)
        and isinstance(lambda_, float)
    ):
        return (((y_hat - y).T @ (y_hat - y))[0][0] + lambda_ * l2(theta)) / (
            2 * y.shape[0]
        )


if __name__ == "__main__":
    y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
    y_hat = np.array([[3], [13], [-11.5], [5], [11], [5], [-20]])
    theta = np.array([[1], [2.5], [1.5], [-0.9]])

    print(reg_loss_(y, y_hat, theta, 0.5))
    # Output: 0.8503571428571429

    print(reg_loss_(y, y_hat, theta, 0.05))
    # Output: 0.5511071428571429

    print(reg_loss_(y, y_hat, theta, 0.9))
    # Output: 1.116357142857143

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


def vec_log_loss_(y, y_hat, eps=1e-15):
    """Compute the logistic loss value.
    Args:
    y: has to be an numpy.array, a vector of shape m * 1.
    y_hat: has to be an numpy.array, a vector of shape m * 1.
    eps: epsilon (default=1e-15)
    Return:
    The logistic loss value as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    if check_vector(y) and check_vector(y_hat) and y.shape == y_hat.shape:
        one = np.ones(y.shape)
        return (
            -(y.T @ np.log(y_hat + eps) + (one - y).T @ np.log(one - y_hat + eps))[0][0]
            / y.shape[0]
        )

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



def reg_log_loss_(y, y_hat, theta, lambda_, eps=1e-15):
    """Computes the regularized loss of a logistic regression model from two non-empty numpy.array,
    without any for loop. The two arrays must have the same shapes.
    Args:
    y: has to be an numpy.array, a vector of shape m * 1.
    y_hat: has to be an numpy.array, a vector of shape m * 1.
    theta: has to be a numpy.array, a vector of shape n * 1.
    lambda_: has to be a float.
    eps: has to be a float, epsilon (default=1e-15).
    Return:
    The regularized loss as a float.
    None if y, y_hat, or theta is empty numpy.array.
    None if y or y_hat have component outside [0 ; 1]
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
        and isinstance(eps, float)
    ):
        return vec_log_loss_(y, y_hat, eps) + (lambda_ * l2(theta)) / (2 * y.shape[0])


if __name__ == "__main__":
    y = np.array([[1], [1], [0], [0], [1], [1], [0]])
    y_hat = np.array([[0.9], [0.79], [0.12], [0.04], [0.89], [0.93], [0.01]])
    theta = np.array([[1], [2.5], [1.5], [-0.9]])

    print(reg_log_loss_(y, y_hat, theta, 0.5))
    # Output: 0.40824105118138265

    print(reg_log_loss_(y, y_hat, theta, 0.05))
    # Output: 0.10899105118138264

    print(reg_log_loss_(y, y_hat, theta, 0.9))
    # Output: 0.6742410511813826

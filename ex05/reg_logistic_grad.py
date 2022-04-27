#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

import numpy as np


def check_matrix(matrix):
    if isinstance(matrix, np.ndarray) and matrix.size != 0 and len(matrix.shape) == 2:
        return True
    exit("Error matrix")


def check_vector(vector):
    if (
        isinstance(vector, np.ndarray)
        and vector.size != 0
        and len(vector.shape) == 2
        and vector.shape[1] == 1
    ):
        return True
    exit("Error vector")


def sigmoid_(x):
    """
    Compute the sigmoid of a vector.
    Args:
    x: has to be an numpy.array, a vector
    Return:
    The sigmoid value as a numpy.array.
    None otherwise.
    Raises:
    This function should not raise any Exception.
    """
    if isinstance(x, np.ndarray) and x.size != 0 and x.shape == ():
        return [[1 / (1 + np.exp(-x))]]
    if check_vector(x):
        return 1 / (1 + np.exp(-x))


def reg_logistic_grad(y, x, theta, lambda_):
    """Computes the regularized logistic gradient of three non-empty numpy.array,
    with two for-loops. The three arrays must have compatible shapes.
    Args:
    y: has to be a numpy.array, a vector of shape m * 1.
    x: has to be a numpy.array, a matrix of dimesion m * n.
    theta: has to be a numpy.array, a vector of shape (n + 1) * 1.
    lambda_: has to be a float.
    Return:
    A numpy.array, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
    None if y, x, or theta are empty numpy.array.
    None if y, x or theta does not share compatibles shapes.
    None if y, x or theta or lambda_ is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    try:
        lambda_ = float(lambda_)
    except:
        pass
    if (
        check_vector(y)
        and check_matrix(x)
        and y.shape[0] == x.shape[0]
        and check_vector(theta)
        and theta.shape[0] == x.shape[1] + 1
        and isinstance(lambda_, float)
    ):
        x = np.insert(x, 0, values=1.0, axis=1).astype(float)
        theta_prime = np.array(theta, copy=True)
        theta_prime[0] = 0
        return (x.T @ (sigmoid_(x @ theta) - y) + lambda_ * theta_prime) / x.shape[0]


def vec_reg_logistic_grad(y, x, theta, lambda_):
    """Computes the regularized logistic gradient of three non-empty numpy.array,
    without any for-loop. The three arrays must have compatible shapes.
    Args:
    y: has to be a numpy.array, a vector of shape m * 1.
    x: has to be a numpy.array, a matrix of dimesion m * n.
    theta: has to be a numpy.array, a vector of shape (n + 1) * 1.
    lambda_: has to be a float.
    Return:
    A numpy.array, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
    None if y, x, or theta are empty numpy.array.
    None if y, x or theta does not share compatibles shapes.
    None if y, x or theta or lambda_ is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    try:
        lambda_ = float(lambda_)
    except:
        pass
    if (
        check_vector(y)
        and check_matrix(x)
        and y.shape[0] == x.shape[0]
        and check_vector(theta)
        and theta.shape[0] == x.shape[1] + 1
        and isinstance(lambda_, float)
    ):
        x = np.insert(x, 0, values=1.0, axis=1).astype(float)
        theta_prime = np.array(theta, copy=True)
        theta_prime[0] = 0
        return (x.T @ (sigmoid_(x @ theta) - y) + lambda_ * theta_prime) / x.shape[0]


if __name__ == "__main__":
    x = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    y = np.array([[0], [1], [1]])
    theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])

    print(reg_logistic_grad(y, x, theta, 1))
    # Output: np.array([[-0.55711039], [-1.40334809], [-1.91756886], [-2.56737958], [-3.03924017]])

    print(vec_reg_logistic_grad(y, x, theta, 1))
    # Output: np.array([[-0.55711039], [-1.40334809], [-1.91756886], [-2.56737958], [-3.03924017]])

    print()

    print(reg_logistic_grad(y, x, theta, 0.5))
    # Output: np.array([[-0.55711039], [-1.15334809], [-1.96756886], [-2.33404624], [-3.15590684]])

    print(vec_reg_logistic_grad(y, x, theta, 0.5))
    # Output: np.array([[-0.55711039], [-1.15334809], [-1.96756886], [-2.33404624], [-3.15590684]])

    print()

    print(reg_logistic_grad(y, x, theta, 0.0))
    # Output: np.array([[-0.55711039], [-0.90334809], [-2.01756886], [-2.10071291], [-3.27257351]])

    print(vec_reg_logistic_grad(y, x, theta, 0.0))
    # Output: np.array([[-0.55711039], [-0.90334809], [-2.01756886], [-2.10071291], [-3.27257351]])

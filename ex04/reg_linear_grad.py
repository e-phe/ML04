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


def reg_linear_grad(y, x, theta, lambda_):
    """Computes the regularized linear gradient of three non-empty numpy.array,
    with two for-loop. The three arrays must have compatible shapes.
    Args:
    y: has to be a numpy.array, a vector of shape m * 1.
    x: has to be a numpy.array, a matrix of dimesion m * n.
    theta: has to be a numpy.array, a vector of shape (n + 1) * 1.
    lambda_: has to be a float.
    Return:
    A numpy.array, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
    None if y, x, or theta are empty numpy.array.s
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
        return (x.T @ (x @ theta - y) + lambda_ * theta_prime) / x.shape[0]


def vec_reg_linear_grad(y, x, theta, lambda_):
    """Computes the regularized linear gradient of three non-empty numpy.array,
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
        return (x.T @ (x @ theta - y) + lambda_ * theta_prime) / x.shape[0]


if __name__ == "__main__":
    x = np.array(
        [
            [-6, -7, -9],
            [13, -2, 14],
            [-7, 14, -1],
            [-8, -4, 6],
            [-5, -9, 6],
            [1, -5, 11],
            [9, -11, 8],
        ]
    )
    y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
    theta = np.array([[7.01], [3], [10.5], [-6]])

    print(reg_linear_grad(y, x, theta, 1))
    # Output: np.array([[-60.99], [-195.64714286], [863.46571429], [-644.52142857]])

    print(vec_reg_linear_grad(y, x, theta, 1))
    # Output: np.array([[-60.99], [-195.64714286], [863.46571429], [-644.52142857]])

    print()

    print(reg_linear_grad(y, x, theta, 0.5))
    # Output: np.array([[-60.99], [-195.86142857], [862.71571429], [-644.09285714]])

    print(vec_reg_linear_grad(y, x, theta, 0.5))
    # Output: np.array([[-60.99], [-195.86142857], [862.71571429], [-644.09285714]])

    print()

    print(reg_linear_grad(y, x, theta, 0.0))
    # Output: np.array([[-60.99], [-196.07571429], [861.96571429], [-643.66428571]])

    print(vec_reg_linear_grad(y, x, theta, 0.0))
    # Output: np.array([[-60.99], [-196.07571429], [861.96571429], [-643.66428571]])

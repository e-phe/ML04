#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

import numpy as np


def check_array(array):
    if isinstance(array, list) and len(array) != 0:
        return True
    exit("Error array")


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


class MyLogisticRegression:
    """
    Description:
    My personal logistic regression to classify things.
    """

    def __init__(self, theta, alpha=0.01, max_iter=10000, penalty="l2"):
        if (
            check_array(theta)
            and isinstance(alpha, float)
            and isinstance(max_iter, int)
            and isinstance(penalty, str)
        ):
            self.alpha = alpha
            self.max_iter = max_iter
            self.theta = np.array(theta).reshape(-1, 1)
            self.penalty = penalty
            if penalty == "l2":
                self.lambda_ = 0.5
            elif penalty == "none":
                self.lambda_ = 0
        else:
            return

    def sigmoid_(self, x):
        if isinstance(x, np.ndarray) and x.size != 0 and x.shape == ():
            return [[1 / (1 + np.exp(-x))]]
        if check_vector(x):
            return 1 / (1 + np.exp(-x))

    def predict_(self, x):
        if (
            check_matrix(x)
            and check_vector(self.theta)
            and x.shape[1] + 1 == self.theta.shape[0]
        ):
            x = np.insert(x, 0, values=1.0, axis=1).astype(float)
            return self.sigmoid_(x @ self.theta)

    def loss_elem_(self, x, y):
        eps = 1e-15
        y_hat = self.predict_(x)
        if check_vector(y) and check_vector(y_hat) and y.shape == y_hat.shape:
            one = np.ones(y.shape)
            return (
                -(y.T @ np.log(y_hat + eps) + (one - y).T @ np.log(one - y_hat + eps))
            )[0][0]

    def vec_log_loss_(self, y, y_hat, eps=1e-15):
        if (
            check_vector(y)
            and check_vector(y_hat)
            and y.shape == y_hat.shape
            and isinstance(eps, float)
        ):
            one = np.ones(y.shape)
            return (
                -(y.T @ np.log(y_hat + eps) + (one - y).T @ np.log(one - y_hat + eps))[
                    0
                ][0]
                / y.shape[0]
            )

    def l2(self, theta):
        if check_vector(theta):
            ret = theta[1:].T @ theta[1:]
            return ret[0][0]

    def loss_(self, x, y):
        y_hat = self.predict_(x)
        if check_vector(y) and check_vector(y_hat) and y.shape == y_hat.shape:
            return self.vec_log_loss_(y, y_hat) + (
                self.lambda_ * self.l2(self.theta)
            ) / (2 * y.shape[0])

    def fit_(self, x, y):
        if (
            check_matrix(x)
            and check_vector(y)
            and check_vector(self.theta)
            and x.shape[0] == y.shape[0]
            and x.shape[1] + 1 == self.theta.shape[0]
        ):
            x = np.insert(x, 0, values=1.0, axis=1).astype(float)
            theta_prime = np.array(self.theta, copy=True)
            theta_prime[0] = 0
            for _ in range(self.max_iter):
                gradient = (
                    x.T @ (self.sigmoid_(x @ self.theta) - y)
                    + self.lambda_ * theta_prime
                ) / x.shape[0]
                self.theta -= self.alpha * gradient
            return self.theta


if __name__ == "__main__":
    X = np.array([[1.0, 1.0, 2.0, 3.0], [5.0, 8.0, 13.0, 21.0], [3.0, 5.0, 9.0, 14.0]])
    Y = np.array([[1], [0], [1]])
    mylr = MyLogisticRegression([2, 0.5, 7.1, -4.3, 2.09])

    print(mylr.predict_(X))
    print(mylr.loss_(X, Y))
    print(mylr.fit_(X, Y))
    print(mylr.predict_(X))
    print(mylr.loss_(X, Y))

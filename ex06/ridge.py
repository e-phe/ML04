#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

import numpy as np
import pprint

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


class MyRidge:
    """
    Description:
    My personal ridge regression class to fit like a boss.
    """

    def __init__(self, thetas, alpha=0.001, max_iter=1000, lambda_=0.5):
        if (
            check_vector(thetas)
            and isinstance(alpha, float)
            and isinstance(max_iter, int)
            and isinstance(lambda_, float)
        ):
            self.alpha = alpha
            self.max_iter = max_iter
            self.thetas = thetas
            self.lambda_ = lambda_
        else:
            exit("Error")

    def get_params_(self):
        return self.__dict__

    def set_params_(self, **kwargs):
        self.__dict__.update(kwargs)
        return type(self).__name__ + pprint.pformat(self.__dict__)

    def l2(self):
        if check_vector(self.thetas):
            ret = self.thetas[1:].T @ self.thetas[1:]
            return ret[0][0]

    def loss_(self, y, y_hat):
        if (
            check_vector(y)
            and check_vector(y_hat)
            and y.shape == y_hat.shape
            and isinstance(self.lambda_, float)
        ):
            return (((y_hat - y).T @ (y_hat - y))[0][0] + self.lambda_ * self.l2()) / (
                2 * y.shape[0]
            )

    def loss_elem_(self, y, y_hat):
        if check_vector(y) and check_vector(y_hat) and y.shape == y_hat.shape:
            return np.array(
                [(y_hat[i] - y[i]) * (y_hat[i] - y[i]) for i in range(y.shape[0])]
            )
        return

    def predict_(self, x):
        if isinstance(x, np.ndarray) and x.size != 0:
            x = np.insert(x, 0, values=1.0, axis=1).astype(float)
            return x @ self.thetas
        return

    def gradient_(self, x, y):
        if (
            check_vector(y)
            and check_matrix(x)
            and y.shape[0] == x.shape[0]
            and check_vector(self.thetas)
            and self.thetas.shape[0] == x.shape[1] + 1
            and isinstance(self.lambda_, float)
        ):
            x = np.insert(x, 0, values=1.0, axis=1).astype(float)
            theta_prime = np.array(self.thetas, copy=True)
            theta_prime[0] = 0
            return (x.T @ (x @ self.thetas - y) + self.lambda_ * theta_prime) / x.shape[
                0
            ]

    def fit_(self, x, y):
        if (
            check_vector(y)
            and check_matrix(x)
            and y.shape[0] == x.shape[0]
            and check_vector(self.thetas)
            and self.thetas.shape[0] == x.shape[1] + 1
            and isinstance(self.lambda_, float)
        ):
            x = np.insert(x, 0, values=1.0, axis=1).astype(float)
            theta_prime = np.array(self.thetas, copy=True)
            theta_prime[0] = 0
            for _ in range(self.max_iter):
                gradient = (
                    x.T @ (x @ self.thetas - y) + self.lambda_ * theta_prime
                ) / x.shape[0]
                self.thetas = self.thetas - self.alpha * gradient
            return self.thetas


from sklearn.linear_model import Ridge

if __name__ == "__main__":
    y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
    y_hat = np.array([[3], [13], [-11.5], [5], [11], [5], [-20]])
    theta = np.array([[1], [2.5], [1.5], [-0.9]])
    mr = MyRidge(theta)
    rr = Ridge(theta)

    print("get_params")
    print(mr.get_params_())

    print("\nset_params")
    print(mr.set_params_(max_iter=1000))
    print(rr.set_params(max_iter=1000))

    print("\nloss")

    print(mr.loss_(y, y_hat))
    # Output: 0.8503571428571429

    mr.lambda_ = 0.05
    print(mr.loss_(y, y_hat))
    # Output: 0.5511071428571429

    mr.lambda_ = 0.9
    print(mr.loss_(y, y_hat))
    # Output: 1.116357142857143

    print("\nloss_elem_")
    print(mr.loss_elem_(y, y_hat))

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
    mr.thetas = np.array([[7.01], [3], [10.5], [-6]])

    print("\npredict")
    print(mr.predict_(x))

    print("\ngradient")

    mr.lambda_ = 1.0
    print(mr.gradient_(x, y))
    # Output: np.array([[-60.99], [-195.64714286], [863.46571429], [-644.52142857]])

    mr.lambda_ = 0.5
    print(mr.gradient_(x, y))
    # Output: np.array([[-60.99], [-195.86142857], [862.71571429], [-644.09285714]])

    mr.lambda_ = 0.0
    print(mr.gradient_(x, y))
    # Output: np.array([[-60.99], [-196.07571429], [861.96571429], [-643.66428571]])

    print("\nfit")
    print(mr.fit_(x, y))

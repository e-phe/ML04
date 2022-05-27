#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

import numpy as np
import os
import pandas as pd


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


def data_spliter(x, y, proportion):
    if (
        check_matrix(x)
        and check_vector(y)
        and x.shape[0] == y.shape[0]
        and isinstance(proportion, float)
        and proportion <= 1
    ):
        df = np.hstack((x, y))
        np.random.shuffle(df)
        x_train, x_test = np.split(df[:, :-1], [int(proportion * x.shape[0])])
        y_train, y_test = np.split(df[:, [-1]], [int(proportion * y.shape[0])])
        return (x_train, x_test, y_train, y_test)
    return


def zscore(x):
    if check_vector(x):
        mean = sum(x) / x.shape[0]
        std = np.sqrt(np.square(x - mean).sum() / x.shape[0])
        x = (x - mean) / std
        return x


class MyLogisticRegression:
    """
    Description:
    My personal logistic regression to classify things.
    """

    def __init__(self, theta, alpha=0.01, max_iter=10000, lambda_=0.5):
        if (
            check_vector(theta)
            and isinstance(alpha, float)
            and isinstance(max_iter, int)
        ):
            self.alpha = alpha
            self.max_iter = max_iter
            self.theta = np.array(theta).reshape(-1, 1)
            self.lambda_ = lambda_
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

    def precision_score_(self, y, y_hat, pos_label=1):
        if (
            check_vector(y)
            and check_vector(y_hat)
            and y.shape == y_hat.shape
            and (isinstance(pos_label, str) or isinstance(pos_label, int))
        ):
            pos = y == pos_label
            pos_hat = y_hat == pos_label
            unique, tp = np.unique(pos & pos_hat, return_counts=True)
            unique, all_pos = np.unique(pos_hat, return_counts=True)
            return tp[1] / all_pos[1]

    def recall_score_(self, y, y_hat, pos_label=1):
        if (
            check_vector(y)
            and check_vector(y_hat)
            and y.shape == y_hat.shape
            and (isinstance(pos_label, str) or isinstance(pos_label, int))
        ):
            pos = y == pos_label
            pos_hat = y_hat == pos_label
            unique, tp = np.unique(pos & pos_hat, return_counts=True)
            fn = sum(
                y_hat[i] != pos_label and y[i] != y_hat[i] for i in range(y.shape[0])
            )
            return tp[1] / (tp[1] + int(fn))

    def f1_score_(self, y, y_hat, pos_label=1):
        if (
            check_vector(y)
            and check_vector(y_hat)
            and y.shape == y_hat.shape
            and (isinstance(pos_label, str) or isinstance(pos_label, int))
        ):
            return (
                2
                * self.precision_score_(y, y_hat, pos_label)
                * self.recall_score_(y, y_hat, pos_label)
            ) / (
                self.precision_score_(y, y_hat, pos_label)
                + self.recall_score_(y, y_hat, pos_label)
            )


if __name__ == "__main__":
    try:
        if os.stat("../resources/solar_system_census.csv").st_size > 0:
            data = np.loadtxt(
                "../resources/solar_system_census.csv",
                dtype=float,
                delimiter=",",
                skiprows=1,
            )
            planets = np.loadtxt(
                "../resources/solar_system_census_planets.csv",
                dtype=float,
                delimiter=",",
                skiprows=1,
            )
        else:
            exit("FileNotFoundError")
    except:
        exit("FileNotFoundError")

    df = np.hstack((data[:, 1:], planets[:, 1:]))
    for i in range(df.shape[1] - 1):
        df[:, [i]] = zscore(df[:, [i]])
    (x, x_validation, y, y_validation) = data_spliter(df[:, :-1], df[:, [-1]], 0.7)

    thetas = []
    pd_thetas = []
    for j in range(6):
        for zipcode in range(4):
            mlr = MyLogisticRegression(
                np.ones(x.shape[1] + 1).reshape(-1, 1), lambda_=j / 5
            )
            y_ones = np.where(y == zipcode, 1, 0)
            theta = mlr.fit_(x, y_ones)
            thetas.append(theta)
            pd_thetas.append(theta.T[0])

        x_pred = np.insert(x, 0, values=1.0, axis=1).astype(float)
        y_hat = np.array(
            [
                max((i @ np.array(thetas[zipcode]), zipcode) for zipcode in range(4))[1]
                for i in x_pred
            ]
        ).reshape(-1, 1)
        print("lambda =", j / 5)
        print(mlr.f1_score_(y, y_hat))

        x_pred_validation = np.insert(x_validation, 0, values=1.0, axis=1).astype(float)
        y_hat_validation = np.array(
            [
                max((i @ np.array(thetas[zipcode]), zipcode) for zipcode in range(4))[1]
                for i in x_pred_validation
            ]
        ).reshape(-1, 1)
        print(mlr.f1_score_(y_validation, y_hat_validation))

    pd.DataFrame(pd_thetas).to_csv("models.csv", index=None)

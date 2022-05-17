#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

from ridge import MyRidge
import matplotlib.pyplot as plt
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
        np.random.shuffle(x)
        np.random.shuffle(y)
        x_train, x_test = np.split(x, [int(proportion * x.shape[0])])
        y_train, y_test = np.split(y, [int(proportion * y.shape[0])])
        return (x_train, x_test, y_train, y_test)
    return


def add_polynomial_features(x, power):
    if check_vector(x) and isinstance(power, int):
        return np.vander(x.reshape(1, -1)[0], power + 1, increasing=True)[:, 1:]
    return


def vander_matrix(x, y):
    x_ = np.zeros(y.shape)
    for x_train in x.T:
        x_ = np.concatenate(
            (x_, add_polynomial_features(x_train.reshape(-1, 1), i)), axis=1
        )
    return x_[:, 1:]


if __name__ == "__main__":
    try:
        if os.stat("../resources/space_avocado.csv").st_size > 0:
            data = np.loadtxt(
                "../resources/space_avocado.csv",
                dtype=float,
                delimiter=",",
                skiprows=1,
            )
        else:
            exit("FileNotFoundError")
    except:
        exit("FileNotFoundError")

    (x, x_test, y, y_test) = data_spliter(data[:, 1:4], data[:, [4]], 0.7)

    figure, axis = plt.subplots(3, 4)
    alpha = 1
    theta = []
    mse_train = np.zeros((4, 6))
    mse_test = np.zeros((4, 6))
    for i in range(1, 5):
        alpha *= 1e-7
        for j in range(6):
            x_ = vander_matrix(x, y)
            my_lr = MyRidge(
                np.ones(x_.shape[1] + 1).reshape(-1, 1), alpha, 10000, j / 5
            )
            if j == 0:
                y_hat = np.empty(y.shape + (6,))

            theta.append(my_lr.fit_(x_, y).T[0])
            y_hat[:, :, j] = my_lr.predict_(x_)

            x_test_ = vander_matrix(x_test, y_test)
            mse_train[i - 1, j] = my_lr.mse_(x_, y)
            mse_test[i - 1, j] = my_lr.mse_(x_test_, y_test)
        x_name = [
            "weight(in ton)",
            "prod_distance (in Mkm)",
            "time_delivery (in days)",
        ]
        for j in range(x.shape[1]):
            axis[j, i - 1].set_xlabel(x_name[j])
            axis[j, i - 1].set_ylabel("target (in trantorian unit)")
            axis[j, i - 1].set_ylim([1e5, 1e6])

            axis[j, i - 1].scatter(x[:, j], y, label="dataset_train")
            for k in range(6):
                axis[j, i - 1].scatter(
                    x[:, j],
                    y_hat[:, :, k],
                    marker=".",
                    label="prediction lambda {}".format(k / 5),
                )
            axis[j, i - 1].legend()
    plt.show()

    plt.xlabel("polynomial degree")
    plt.ylabel("mse")
    for j in range(6):
        plt.plot(
            np.arange(1, 5), mse_train[:, j], label="mse_train lambda {}".format(j / 5)
        )
        plt.scatter(
            np.arange(1, 5), mse_train[:, j], label="mse_train lambda {}".format(j / 5)
        )
        plt.plot(
            np.arange(1, 5), mse_test[:, j], label="mse_test lambda {}".format(j / 5)
        )
        plt.scatter(
            np.arange(1, 5),
            mse_test[:, j],
            marker=".",
            label="mse_test lambda {}".format(j / 5),
        )
    plt.legend()
    plt.show()

    pd.DataFrame(theta).to_csv("models.csv", index=None)

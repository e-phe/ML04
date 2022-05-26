#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

from ridge import MyRidge
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


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


def vander_matrix(x, y, power):
    if (
        check_matrix(x)
        and check_vector(y)
        and x.shape[0] == y.shape[0]
        and isinstance(power, int)
    ):
        x_ = np.zeros(y.shape)
        for x_train in x.T:
            x_ = np.concatenate(
                (x_, add_polynomial_features(x_train.reshape(-1, 1), power)), axis=1
            )
        return x_[:, 1:]


if __name__ == "__main__":
    try:
        if (
            os.stat("../resources/space_avocado.csv").st_size > 0
            and os.stat("models.csv").st_size > 0
        ):
            data = np.loadtxt(
                "../resources/space_avocado.csv",
                dtype=float,
                delimiter=",",
                skiprows=1,
            )
            models = np.genfromtxt("models.csv", dtype=str, delimiter="\t")
        else:
            exit("FileNotFoundError")
    except:
        exit("FileNotFoundError")

    (x, x_test, y, y_test) = data_spliter(data[:, 1:4], data[:, [4]], 0.9)
    (x, x_validation, y, y_validation) = data_spliter(x, y, 0.7)

    theta = []
    for el in models:
        theta.append(np.fromstring(el, sep=","))
    theta = theta[1:]

    mse_train = np.zeros((4, 6))
    mse_validation = np.zeros((4, 6))
    count = 0
    for i in range(1, 5):
        for j in range(6):
            x_ = vander_matrix(x, y, i)
            x_validation_ = vander_matrix(x_validation, y_validation, i)
            my_lr = MyRidge(theta[j + count].reshape(-1, 1), 1e-7, 10000, j / 5)

            mse_train[i - 1, j] = my_lr.mse_(x_, y)
            mse_validation[i - 1, j] = my_lr.mse_(x_validation_, y_validation)
        count += 6

    diff = abs(mse_train - mse_validation)
    (degree, lambda_) = np.where(diff == np.min(diff))
    (degree, lambda_) = (int(degree[0] + 1), int(lambda_[0]))
    print("best hypothesis: degree", degree, "lambda", lambda_ / 5)

    x_ = vander_matrix(x, y, degree)
    my_lr = MyRidge(
        np.ones(x_.shape[1] + 1).reshape(-1, 1),
        pow(1e-7, degree),
        10000,
        lambda_=lambda_ / 5,
    )
    my_lr.fit_(x_, y)

    x_test_ = vander_matrix(x_test, y_test, degree)
    print("mse of the best model", my_lr.mse_(x_test_, y_test))

    plt.xlabel("polynomial degree")
    plt.ylabel("mse")
    for j in range(6):
        plt.plot(
            np.arange(1, 5), mse_train[:, j], label="mse_train lambda {}".format(j / 5)
        )
        plt.scatter(np.arange(1, 5), mse_train[:, j])
        plt.plot(
            np.arange(1, 5),
            mse_validation[:, j],
            label="mse_test lambda {}".format(j / 5),
        )
        plt.scatter(np.arange(1, 5), mse_validation[:, j])
    plt.legend()
    plt.show()

    x_name = [
        "weight(in ton)",
        "prod_distance (in Mkm)",
        "time_delivery (in days)",
    ]
    figure, axis = plt.subplots(1, 3)
    for i in range(x.shape[1]):
        axis[i].set_xlabel(x_name[i])
        axis[i].set_ylabel("target (in trantorian unit)")
        axis[i].set_ylim([1e5, 1e6])

        axis[i].scatter(x[:, i], y, label="dataset_train")
        for j in range(6):
            x_ = vander_matrix(x, y, degree)
            my_lr = MyRidge(
                theta[(degree - 1) * 6 + lambda_].reshape(-1, 1), lambda_=j / 5
            )
            y_hat = my_lr.predict_(x)

            axis[i].scatter(
                x[:, i],
                y_hat,
                marker=".",
                label="prediction lambda {}".format(j / 5),
            )
        axis[i].legend()
    plt.show()

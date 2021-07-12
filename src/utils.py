import h5py
import numpy as np
from sklearn.preprocessing import OneHotEncoder

import datasets.mnist.loader as mnist


def load_data_cat():
    train_dataset = h5py.File("datasets/catvnoncat/train_catvnoncat.h5", "r")
    X_train = np.array(train_dataset["train_set_x"][:])  # your train set features
    y_train = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File("datasets/catvnoncat/test_catvnoncat.h5", "r")
    X_test = np.array(test_dataset["test_set_x"][:])  # your test set features
    y_test = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    # Reshape the training and test examples
    X_train = X_train.reshape(X_train.shape[0], -1).T / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1).T / 255.0
    y_train = y_train.reshape((1, y_train.shape[0]))
    y_test = y_test.reshape((1, y_test.shape[0]))

    return X_train, y_train, X_test, y_test, classes


def load_data_mnist():
    X_train, y_train, X_test, y_test = mnist.get_data()
    # Reshape the training and test examples
    X_train = X_train.reshape(X_train.shape[0], -1).T / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1).T / 255.0

    enc = OneHotEncoder(sparse=False, categories="auto")
    y_train = enc.fit_transform(y_train.reshape(len(y_train), -1)).T
    y_test = enc.transform(y_test.reshape(len(y_test), -1)).T

    return X_train, y_train, X_test, y_test

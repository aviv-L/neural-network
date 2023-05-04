import numpy as np
import tensorflow as tf
import json


def load_mnist_train():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()  # load mnist dataset
    num_classes = 10  # number of output classes

    y_train_oneHot = np.zeros((y_train.size, num_classes))  # create one-hot encoded labels
    y_train_oneHot[np.arange(y_train.size), y_train] = 1  # set the corresponding class for each sample
    x_train = x_train.reshape((x_train.shape[0], -1))  # flatten the input images

    x_train = x_train / 255.0  # normalize the input

    input_size = 784  # input layer size
    hidden_size = 64  # hidden layer size
    output_size = 10  # output layer size

    return x_train, y_train_oneHot, input_size, hidden_size, output_size


def load_mnist_test():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()  # load mnist dataset

    num_classes = 10  # number of output classes

    y_test_oneHot = np.zeros((y_test.size, num_classes))  # create one-hot encoded labels
    y_test_oneHot[np.arange(y_test.size), y_test] = 1  # set the corresponding class for each sample

    x_test_img = x_test  # save images for visualization later
    x_test = x_test.reshape((x_test.shape[0], -1))  # flatten the input images

    x_test = x_test / 255.0  # normalize the input

    input_size = 784  # input layer size
    hidden_size = 64  # hidden layer size
    output_size = 10  # output layer size

    with open("weights1.json", 'r') as r:
        weights = json.load(r)  # load the weights saved in json format

    return x_test, y_test_oneHot, input_size, hidden_size, output_size, weights, x_test_img

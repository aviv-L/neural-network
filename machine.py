import numpy as np


# Define sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Define forward function
def forward(x, w1, w2, b1, b2):
    # Calculate first layer of activations
    z1 = np.dot(x, w1) + b1
    a1 = sigmoid(z1)
    # Calculate second layer of activations (output)
    z2 = np.dot(a1, w2) + b2
    # Apply softmax function to get probability distribution
    q = np.exp(z2) / np.sum(np.exp(z2), axis=1, keepdims=True)
    return q, a1, z1


# Define loss function
def compute_loss(y, y_hat):
    # Calculate mean squared error loss
    loss = 0.5 * np.sum((y_hat - y) ** 2)
    return loss


# Define sigmoid derivative function
def sigmoid_derivative(k):
    # Calculate derivative of sigmoid function
    s = sigmoid(k) * (1 - sigmoid(k))
    return s


# Define backward function for computing gradients
def backward(x, y, y_hat, a1, z1, w2):
    # Reshape input for correct dimensions
    x = x.reshape(1, -1)
    # Get number of training examples
    m = y.shape[0]
    # Compute gradients for second layer
    dz2 = y_hat - y
    dz2 = dz2.reshape(1, -1)
    db1 = np.dot(a1.T, dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m
    # Compute gradients for first layer
    da1 = np.dot(dz2, w2.T)
    dz1 = da1 * sigmoid_derivative(z1)
    dw1 = np.dot(x.T, dz1) / m
    dw2 = np.sum(dz1, axis=0, keepdims=True) / m
    return dw1, db1, dw2, db2

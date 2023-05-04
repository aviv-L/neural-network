import numpy as np
import machine as m
import json
import data_loader as dl


def train():

    # Load MNIST training data
    x_train, y_train_oneHot, input_size, hidden_size, output_size = dl.load_mnist_train()

    # Initialize weights and biases
    w1 = np.random.rand(input_size, hidden_size) * 0.1  # Randomly initialize weights for input to hidden layer
    w2 = np.random.rand(hidden_size, output_size) * 0.1  # Randomly initialize weights for hidden to output layer
    b1 = np.zeros((1, hidden_size))  # Initialize biases for hidden layer as zeros
    b2 = np.zeros((1, output_size))  # Initialize biases for output layer as zeros

    y_h = np.zeros(y_train_oneHot.shape)
    learning_rate = 0.001
    epochs = 400
    prev_loss = float('inf')  # Initialize previous loss to a very large value
    for epoch in range(epochs):
        for sample in range(y_train_oneHot.shape[0]):
            # Forward pass
            y_h[sample], a1, z1 = m.forward(x_train[sample], w1, w2, b1, b2)
            # Calculate loss
            loss = m.compute_loss(y_train_oneHot[sample], y_h[sample])
            # Backward pass to compute gradients
            dw1, dw2, db1, db2 = m.backward(x_train[sample], y_train_oneHot[sample], y_h[sample], a1, z1, w2)

            # Update weights and biases using gradient descent
            w1 -= learning_rate * dw1
            w2 -= learning_rate * dw2
            b1 -= learning_rate * db1
            b2 -= learning_rate * db2.reshape(1, 10)

            if sample % 1000 == 0:
                print(f"epoch {epoch}, sample {sample}, loss {loss:.4f}")

        # Calculate average loss for the epoch
        avg_loss = m.compute_loss(y_train_oneHot, y_h) / y_train_oneHot.shape[0]
        print(f"epoch {epoch}, avg_loss {avg_loss:.4f}")

        # Check for convergence
        if abs(prev_loss - avg_loss) < 0.0001:
            print(f"Converged after {epoch + 1} epochs")
            break

        prev_loss = avg_loss

    # Store trained weights and error
    weights = {
        "w1": w1.tolist(),
        "w2": w2.tolist(),
        "b1": b1.tolist(),
        "b2": b2.tolist(),
    }
    error = {
        "error": avg_loss
    }
    data = {"weights": weights, "error": error}
    filename = "weights.json"

    # Save weights and error to JSON file
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)


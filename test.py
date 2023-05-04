import numpy as np
import data_loader as dl
import machine as m
import matplotlib.pyplot as plt


def test():

    # Load test dataset
    x_test, y_test_oneHot, input_size, hidden_size, output_size, weights, x_test_img = dl.load_mnist_test()

    # Initialize weights and biases
    w1 = np.random.rand(input_size, hidden_size)  # weight matrix from input to hidden layer
    w2 = np.random.rand(hidden_size, output_size)  # weight matrix from hidden to output layer
    b1 = np.zeros((1, hidden_size))  # bias vector for hidden layer
    b2 = np.zeros((1, output_size))  # bias vector for output layer

    # Load trained weights and biases from JSON file
    w1 = np.array(weights["weights"]["w1"])
    w2 = np.array(weights["weights"]["w2"])
    b1 = np.array(weights["weights"]["b1"])
    b2 = np.array(weights["weights"]["b2"])

    y_h = np.zeros(y_test_oneHot.shape) # Initialize predicted output matrix
    observations = y_test_oneHot.shape[0] # Calculate number of observations
    i = 0 # Initialize counter for correct predictions

    # Iterate through each observation in test dataset
    for observation in range(observations):

        # Compute predicted output, and store it in y_h matrix
        y_h[observation], a1, z1 = m.forward(x_test[observation], w1, w2, b1, b2)

        # Find the index of the maximum value in predicted output (y_h) and true output (y_test_oneHot)
        y_h_argmax = np.argmax(y_h[observation])
        y_argmax = np.argmax(y_test_oneHot[observation])

        # Increment counter if predicted label is same as true label
        if y_h_argmax == y_argmax:
            i += 1

        # Display the image, and its true and predicted label
        plt.imshow(x_test_img[observation], cmap='gray')
        plt.title(f'True Label: {y_argmax}, Predicted Label: {y_h_argmax}')
        plt.show()

        # Wait for user input before proceeding to the next observation
        input("press anything")

    # Calculate accuracy of the model
    accuracy = (i / y_test_oneHot.shape[0])*100

    # Print the accuracy of the model
    print(f"accuracy: {accuracy}%")
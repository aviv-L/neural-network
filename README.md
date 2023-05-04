# neural-network

The project involves building a neural network to classify handwritten digits from the MNIST dataset. The code is divided into three files:

data_loader.py: Contains functions to load the MNIST dataset, preprocess it, and return the training and testing data.

machine.py: Contains functions to perform the forward and backward propagation steps in the neural network.

test.py: Contains the main function to test the trained neural network on the test set and display the predictions.

The data_loader.py file loads the MNIST dataset using the TensorFlow library and preprocesses- 
-the data by reshaping the images and converting the labels into one-hot vectors.
The preprocessed data is returned to the main function in test.py.

The machine.py file contains the forward() and backward() functions that perform the forward and backward propagation steps,-
-respectively, in the neural network. These functions use the weights and biases passed as arguments to compute the activations and output of the network.

The main function in test.py loads the preprocessed data and the weights of the trained neural network from a JSON file.
It then uses the forward() function to compute the predictions for each test image and compares them to the ground truth labels.
The predicted labels are displayed alongside the corresponding images using the matplotlib library.
The accuracy of the model on the test set is also computed and displayed at the end of the execution.

Overall, the project demonstrates the implementation of a neural network using the mean square error loss function-
-for image classification and its usage for predicting the labels of handwritten digits.
After training for 400 epochs, the model achieved an accuracy of almost 95%.

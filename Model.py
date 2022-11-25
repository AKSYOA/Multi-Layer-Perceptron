import numpy as np

Weights = []
nodes_output = []
errors = []


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def activationFunction(x, activation_function):
    if activation_function == 'Sigmoid':
        return sigmoid(x)
    else:
        return np.tanh(x)


def Train(X, Y, number_of_hidden_layers, number_of_neurons, learning_rate, number_of_epochs, bias_value,
          activation_function_type):
    initializeWeight(number_of_hidden_layers, number_of_neurons)
    for i in range(number_of_epochs):
        for j in range(X.shape[0]):
            feedForward(X[j], number_of_hidden_layers, activation_function_type)


def initializeWeight(number_of_hidden_layers, number_of_neurons):
    number_of_neurons.insert(0, 5)  # 5 neurons (input Layer)
    number_of_neurons.append(3)  # 3 neurons (output Layer)
    for i in range(number_of_hidden_layers + 1):
        Weights.append(np.random.rand(number_of_neurons[i + 1], number_of_neurons[i]))


def feedForward(X_sample, number_of_hidden_layers, activation_function_type):
    # first layer
    X_sample = X_sample.reshape(5, 1)
    F = np.dot(Weights[0], X_sample)
    nodes_output.append(activationFunction(F, activation_function_type))

    # rest of layers
    for i in range(number_of_hidden_layers):
        F = np.dot(Weights[i + 1], nodes_output[i])
        nodes_output.append(activationFunction(F, activation_function_type))

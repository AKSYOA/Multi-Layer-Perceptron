import numpy as np

Weights = []
nodes_output = []
errors = []


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def Train(X, Y, number_of_hidden_layers, number_of_neurons, learning_rate, number_of_epochs, bias_value,
          activation_function):
    initializeWeight(number_of_hidden_layers, number_of_neurons)
    for i in range(number_of_epochs):
        for j in range(X.shape[0]):
            feedForward(X[j], Y[j], number_of_hidden_layers, number_of_neurons, activation_function)


def initializeWeight(number_of_hidden_layers, number_of_neurons):
    number_of_neurons.insert(0, 5)  # 5 neurons (input Layer)
    number_of_neurons.append(3)  # 3 neurons (output Layer)
    for i in range(number_of_hidden_layers + 1):
        Weights.append(np.random.rand(number_of_neurons[i + 1], number_of_neurons[i]))


def feedForward(X_sample, Y_sample, number_of_hidden_layers, number_of_neurons, activation_function):
    # first layer
    X_sample = X_sample.reshape(5, 1)
    F = np.dot(Weights[0], X_sample)
    print(F)
    if activation_function == 'Sigmoid':
        nodes_output.append(sigmoid(F))
    else:
        nodes_output.append(np.tan(F))

    print(nodes_output[0])

    # rest of layers
    for i in range(number_of_hidden_layers):
        F = np.dot(Weights[i + 1], nodes_output[i])
        if activation_function == 'Sigmoid':
            nodes_output.append(sigmoid(F))
        else:
            nodes_output.append(np.tan(F))

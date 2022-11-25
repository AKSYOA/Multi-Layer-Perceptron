import numpy as np

Weights = []


def Train(X, Y, number_of_hidden_layers, number_of_neurons, learning_rate, number_of_epochs, bias_value,
          activation_function):
    initializeWeight(number_of_hidden_layers, number_of_neurons)


def initializeWeight(number_of_hidden_layers, number_of_neurons):
    number_of_neurons.insert(0, 5)  # 5 neurons (input Layer)
    number_of_neurons.append(3)  # 3 neurons (output Layer)
    for i in range(number_of_hidden_layers + 1):
        Weights.append(np.random.rand(number_of_neurons[i + 1], number_of_neurons[i]))


def feedForward():
    print('hello world')

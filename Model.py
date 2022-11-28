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


def clearLists():
    nodes_output.clear()
    errors.clear()


def Train(X, Y, number_of_hidden_layers, number_of_neurons, learning_rate, number_of_epochs, bias_value,
          activation_function_type):
    initializeWeight(number_of_hidden_layers, number_of_neurons)

    X_Train = np.vstack((X[:30], X[50:80], X[100:130]))
    Y_Train = np.vstack((Y[:30], Y[50:80], Y[100:130]))

    X_Test = np.vstack((X[30:50], X[80:100], X[130:150]))
    Y_Test = np.vstack((Y[30:50], Y[80:100], Y[130:150]))
    for i in range(number_of_epochs):
        for j in range(X_Train.shape[0]):
            feedForward(X_Train[j], number_of_hidden_layers, activation_function_type)
            backPropagate(Y_Train[j], number_of_hidden_layers, number_of_neurons)
            updateWeights(X_Train[j], number_of_hidden_layers, learning_rate)
            clearLists()
    TestAccuracy(X_Test, Y_Test, number_of_hidden_layers, activation_function_type)


def initializeWeight(number_of_hidden_layers, number_of_neurons):
    number_of_neurons.insert(0, 5)  # 5 neurons (input Layer)
    number_of_neurons.append(3)  # 3 neurons (output Layer)
    for i in range(number_of_hidden_layers + 1):
        Weights.append(np.random.rand(number_of_neurons[i + 1], number_of_neurons[i]))


def feedForward(X_sample, number_of_hidden_layers, activation_function_type):
    # first layer
    X_sample = X_sample.reshape(5, 1)
    F = np.dot(Weights[0], X_sample)
    output = activationFunction(F, activation_function_type)
    nodes_output.append(output)

    # rest of layers
    for i in range(number_of_hidden_layers):
        F = np.dot(Weights[i + 1], nodes_output[i])
        output = activationFunction(F, activation_function_type)
        nodes_output.append(output)


def backPropagate(Y_sample, number_of_hidden_layers, number_of_neurons):
    # Output Layer
    Y_sample = Y_sample.reshape(3, 1)
    F_dash = nodes_output[number_of_hidden_layers] * (1 - nodes_output[number_of_hidden_layers])
    err = (Y_sample - nodes_output[number_of_hidden_layers]) * F_dash
    errors.append(err)

    # Hidden Layers
    for i in range(1, number_of_hidden_layers + 1):
        F_dash = nodes_output[number_of_hidden_layers - i] * (1 - nodes_output[number_of_hidden_layers - i])

        number_of_columns = Weights[number_of_hidden_layers - (i - 1)].shape[1]
        number_of_rows = Weights[number_of_hidden_layers - (i - 1)].shape[0]
        err_sum = np.zeros((number_of_columns, 1))

        for j in range(number_of_columns):
            err_sum[j] = np.sum(Weights[number_of_hidden_layers - (i - 1)][:, j].reshape(number_of_rows, 1) * errors[0])

        errors.insert(0, err_sum * F_dash)


def updateWeights(X_sample, number_of_hidden_layers, learning_rate):
    for i in range(number_of_hidden_layers + 1):
        if i == 0:
            Weights[0] = np.add(Weights[0], (learning_rate * errors[0] * X_sample))
        else:
            rows = nodes_output[i - 1].shape[0]
            cols = nodes_output[i - 1].shape[1]
            Weights[i] = np.add(Weights[i], (learning_rate * errors[i] * nodes_output[i - 1].reshape(cols, rows)))


def TestAccuracy(X_test, Y_test, number_of_hidden_layers, activation_function_type):
    for i in range(X_test.shape[0]):
        nodes_output.clear()
        feedForward(X_test[i], number_of_hidden_layers, activation_function_type)
        print(nodes_output[-1])

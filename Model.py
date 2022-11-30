import numpy as np
from math import sqrt

Weights = []
nodes_output = []
errors = []


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def activationFunction(x, activation_function_type):
    if activation_function_type == 'Sigmoid':
        return sigmoid(x)
    else:
        return np.tanh(x)


def differentialActivationFunction(index, activation_function_type):
    if activation_function_type == 'Sigmoid':
        return nodes_output[index] * (1 - nodes_output[index])
    else:
        return (1 - nodes_output[index]) * (1 + nodes_output[index])


def clearLists():
    nodes_output.clear()
    errors.clear()


def Train(X, Y, number_of_hidden_layers, number_of_neurons, learning_rate, number_of_epochs, bias_value,
          activation_function_type):
    clearLists()
    initializeWeight(number_of_hidden_layers, number_of_neurons)

    X_Train = np.vstack((X[:30], X[50:80], X[100:130]))
    Y_Train = np.vstack((Y[:30], Y[50:80], Y[100:130]))

    X_Test = np.vstack((X[30:50], X[80:100], X[130:150]))
    Y_Test = np.vstack((Y[30:50], Y[80:100], Y[130:150]))
    for i in range(number_of_epochs):
        for j in range(X_Train.shape[0]):
            feedForward(X_Train[j], number_of_hidden_layers, activation_function_type)
            backPropagate(Y_Train[j], number_of_hidden_layers, activation_function_type)
            updateWeights(X_Train[j], number_of_hidden_layers, learning_rate)
            clearLists()
    DataAccuracy(X_Train, Y_Train, number_of_hidden_layers, activation_function_type, 'Training')
    DataAccuracy(X_Test, Y_Test, number_of_hidden_layers, activation_function_type, 'Testing')


def initializeWeight(number_of_hidden_layers, number_of_neurons):
    number_of_neurons.insert(0, 5)  # 5 neurons (input Layer)
    number_of_neurons.append(3)  # 3 neurons (output Layer)
    for i in range(number_of_hidden_layers + 1):
        Weights.append(np.random.randn(number_of_neurons[i + 1], number_of_neurons[i]))


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


def backPropagate(Y_sample, number_of_hidden_layers, activation_function_type):
    # Output Layer
    Y_sample = Y_sample.reshape(3, 1)
    # F_dash = nodes_output[number_of_hidden_layers] * (1 - nodes_output[number_of_hidden_layers])
    F_dash = differentialActivationFunction(number_of_hidden_layers, activation_function_type)
    err = (Y_sample - nodes_output[number_of_hidden_layers]) * F_dash
    errors.append(err)

    # Hidden Layers
    for i in range(1, number_of_hidden_layers + 1):
        # F_dash = nodes_output[number_of_hidden_layers - i] * (1 - nodes_output[number_of_hidden_layers - i])
        F_dash = differentialActivationFunction(number_of_hidden_layers - i, activation_function_type)
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


def DataAccuracy(X, Y, number_of_hidden_layers, activation_function_type, accuracyType):
    y_prediction_class = []
    y_actual_class = []
    for i in range(X.shape[0]):
        nodes_output.clear()
        feedForward(X[i], number_of_hidden_layers, activation_function_type)
        y_prediction, y_actual = Evaluate(nodes_output[-1], Y[i])
        y_prediction_class.append(y_prediction)
        y_actual_class.append(y_actual)
    buildConfusionMatrix(y_prediction_class, y_actual_class, accuracyType)


def Evaluate(output, Y_sample):
    Y_sample = Y_sample.reshape(3, 1)
    return np.argmax(output), np.argmax(Y_sample)


def buildConfusionMatrix(y_prediction_class, y_actual_class, accuracyType):
    matrix = np.zeros((3, 3), dtype=np.int32)

    for i in range(len(y_actual_class)):
        matrix[y_actual_class[i]][y_prediction_class[i]] += 1
    print(matrix)

    correct = np.trace(matrix)
    accuracy = correct / np.sum(matrix) * 100
    print(accuracyType, 'accuracy: {:.2f}'.format(accuracy))

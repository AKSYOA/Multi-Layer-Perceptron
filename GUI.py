from tkinter import ttk
from tkinter import messagebox

import tkinter as tk
import pandas as pd
import numpy as np
from tkinter import *
import Model

dataFrame = pd.read_csv('penguins.csv')

dataFrame['gender'] = dataFrame['gender'].replace(np.NaN, dataFrame['gender'].mode()[0])
dataFrame['gender'] = dataFrame['gender'].map({'male': 0, 'female': 1})

Species = pd.get_dummies(dataFrame['species'])
dataFrame = dataFrame.drop(['species'], axis=1)

X = np.asarray(dataFrame)
Y = np.asarray(Species)

# window Creation
mlp_window = Tk()
mlp_window.title('MLP Model')
mlp_window.geometry("800x200")

# Labels Creation and style
style = ttk.Style()
style.configure("BW.TLabel", foreground="black")

number_of_hidden_layers_label = ttk.Label(text="Number Of Hidden Layers", style="BW.TLabel") \
    .grid(row=0, column=0, pady=5, padx=10)

number_of_neurons_label = ttk.Label(text="Number of Neurons in each Hidden Layer", style="BW.TLabel") \
    .grid(row=0, column=1, pady=5, padx=10)

learning_rate_label = ttk.Label(text="Learning Rate", style="BW.TLabel") \
    .grid(row=0, column=2, pady=5, padx=10)

number_of_epochs_label = ttk.Label(text="Number of Epochs", style="BW.TLabel") \
    .grid(row=0, column=3, pady=5, padx=10)

add_bias_label = ttk.Label(text="Add bias", style="BW.TLabel") \
    .grid(row=0, column=4, pady=5, padx=10)

activation_function_label = ttk.Label(text="Activation Function", style="BW.TLabel") \
    .grid(row=0, column=5, pady=5, padx=10)

# Text Boxes Creation
number_of_hidden_layers = IntVar()
number_of_hidden_layers_textbox = Entry(mlp_window, width=20, textvariable=number_of_hidden_layers) \
    .grid(row=1, column=0, pady=5, padx=5)

number_of_neurons = StringVar()
number_of_neurons_textbox = Entry(mlp_window, width=30, textvariable=number_of_neurons) \
    .grid(row=1, column=1, pady=5, padx=5)

learning_rate = DoubleVar()
learning_rate_textbox = Entry(mlp_window, width=10, textvariable=learning_rate) \
    .grid(row=1, column=2, pady=5, padx=5)

number_of_epochs = IntVar()
number_of_epochs_textbox = Entry(mlp_window, width=13, textvariable=number_of_epochs) \
    .grid(row=1, column=3, pady=5, padx=5)

# bias Checkbox, activation Combobox creation
bias_value = IntVar()
Checkbutton(mlp_window, text="Bias", variable=bias_value) \
    .grid(row=1, column=4, pady=5, padx=8)

activation_function_comboBox = ttk.Combobox(mlp_window, width=20)
activation_function_comboBox['values'] = ('Sigmoid', 'Hyperbolic Tangent Sigmoid')
activation_function_comboBox.grid(row=1, column=5, pady=5, padx=5)


def validateInput():
    if number_of_hidden_layers.get() == 0:
        return False
    if not number_of_neurons.get():
        return False
    if learning_rate.get() == 0:
        return False
    if number_of_epochs.get() == 0:
        return False
    if not activation_function_comboBox.get():
        return False
    return True


def run():
    if validateInput():
        Model.Train(X, Y, number_of_hidden_layers.get(), number_of_neurons.get(), learning_rate.get(),
                    number_of_epochs.get(), bias_value.get(), activation_function_comboBox.get())
    else:
        messagebox.showerror("Error", "Enter Valid Input")


# run Button
run_button = Button(mlp_window, text='Run', width=10, command=run).grid(row=2, column=2, pady=5, padx=8)

mlp_window.mainloop()

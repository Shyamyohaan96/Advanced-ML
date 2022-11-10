import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
import os
import time
import pickle
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.models import Sequential

# opening the file
file = open("hw0_dataset.pkl", "rb")
f_open = pickle.load(file)
file.close()

# separating the attributes and labels
attribute = f_open['ins']
labels = f_open['outs']


def build_model(n_inputs, n_hidden, n_output, activation='elu', lrate=0.1):
    # sequential model
    model = Sequential();
    model.add(InputLayer(input_shape=(n_inputs,)))
    model.add(Dense(n_hidden, use_bias=True, name="hidden", activation=activation))
    model.add(Dense(n_output, use_bias=True, name="output", activation=activation))

    # optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=lrate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
                                   amsgrad=False)

    # compiling the model
    model.compile(loss='mse', optimizer=opt)

    return model


for i in range(10):
    # building a model
    model = build_model(attribute.shape[1], 4, labels.shape[1], activation='tanh')

    # train
    history = model.fit(x=attribute, y=labels, epochs=200, verbose=True)

    # predicting the model
    output = model.predict(attribute)

    if i == 0:
        hist = output

    if i > 0:
        np.append(hist, output)

    file_name = "experiment_" + str(i + 1) + ".csv"
    # saving the predicted output in a file
    np.savetxt(file_name, output, delimiter=',')

    lab = "run_" + str(i + 1)
    # plotting the graph
    plt.plot(history.history['loss'], label=lab)

plt.ylabel('MSE')
plt.xlabel('epochs')
plt.title('Errors vs Epochs for each run')
plt.legend()
# saving the figure for loss vs epochs
plt.savefig("learning_curve.png")
plt.show()
plt.close()

# plotting the histogram
histo_errors = np.abs(hist)
plt.hist(histo_errors, 50)
plt.savefig("histogram_errors.png")
plt.show()
plt.close()






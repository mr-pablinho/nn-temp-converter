# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 14:02:12 2021
CONVERT CELSIUS TO FAHRENHEIT WITH NEURAL NETWORKS
@author: PMR
"""

# %% Transforming Fahrenheit to Celsius 

# import libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# data arrays
degrees_fahr = np.array([-40.0, 14.0, 32.0, 46.0, 59.0, 72.0, 100.0], dtype=float)
degrees_cels = np.array([-40.0, -10.0, 0.0, 8.0, 15.0, 22.0, 38.0], dtype=float)


# layers (using framework Keras)
units = 1           # dimensionality of the output space
input_shape = 1     # create an input layer to insert before the current layer (one input with one neuron)
layer = tf.keras.layers.Dense(units=units, input_shape=[input_shape])

# create a model
model = tf.keras.Sequential([layer])  # sequential for a plain stack of layers with one input and one output

# configure the model for traing
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),  # Adam optimization algorithm to update network weights iteratively
    loss='mean_squared_error'                 # objective function
    )

# train the model for a fixed number of epochs (iterations on a dataset)
print('Training...')
history = model.fit(degrees_cels, degrees_fahr, epochs=1000, verbose=False)
print('Model has been trained!')

# plot training results
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(history.history['loss'])
plt.title('Training history')

# generate an output prediction
input_pred = 212
output_pred = model.predict([input_pred])
print("Prediction: %.2f °C is %.2f °F" % (input_pred, output_pred))
print("Current layer properties:" , end="")
print('weight = %f; bias = %f ' % (layer.get_weights()[0][0][0], layer.get_weights()[1][0]))

# compare with the actual formula
def fahr_to_celsius(celsius):
    res = (celsius * 9/5) + 32
    return res

fahrenheit = fahr_to_celsius(input_pred)
error = np.abs(output_pred - fahrenheit)
print('The absolute error is %f' % (error))



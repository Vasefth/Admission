import app
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import r2_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import InputLayer


admissions_data = pd.read_csv("admissions_data.csv")
#print(admissions_data.head())
#print(admissions_data.describe())
#print(admissions_data.shape)

labels = admissions_data.iloc[:,-1]
features = admissions_data.iloc[:, 1:8]
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.33, random_state = 42)

scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.fit_transform(features_test)

def design_model(data):
  my_model = Sequential()
  input = InputLayer(input_shape = (data.shape[1], ))
  my_model.add(input)
  hidden_layer_1 = layers.Dense(16, activation='relu')
  my_model.add(hidden_layer_1)
  my_model.add(layers.Dropout(0.1))
  hidden_layer_2 = layers.Dense(8, activation = 'relu')
  my_model.add(hidden_layer_2)
  my_model.add(layers.Dropout(0.2))
  my_model.add(Dense(1))

  optimizer = Adam(learning_rate = 0.01)
  my_model.compile(loss = 'mse', metrics = ['mae'], optimizer = optimizer)
  return my_model

model = design_model(features_train_scaled)

stop = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 20)
history = model.fit(features_train_scaled, labels_train.to_numpy(), epochs = 100, batch_size = 8, verbose = 1, validation_split = 0.2, callbacks = [stop])

#Fit and Evaluate the model
#model.fit(features_train_scaled, labels_train, epochs = 100, batch_size = 8, verbose = 1)

res_mse, res_mae = model.evaluate(features_test_scaled, labels_test, verbose = 0)

print(res_mse, res_mae)


## evauate r-squared score
y_pred = model.predict(features_test_scaled)

print(r2_score(labels_test,y_pred))

# plot MAE and val_MAE over each epoch
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['mae'])
ax1.plot(history.history['val_mae'])
ax1.set_title('model mae')
ax1.set_ylabel('MAE')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'validation'], loc='upper left')

# Plot loss and val_loss over each epoch
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('model loss')
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.legend(['train', 'validation'], loc='upper left')

plt.show()

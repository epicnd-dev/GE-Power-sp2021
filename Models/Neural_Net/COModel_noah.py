import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import Activation, Dense
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# set up training and testing data
data_2011 = pd.read_csv(r'../../Data/Train/gt_2011.csv')
train, test = train_test_split(data_2011, test_size=0.15)
train, val = train_test_split(train, test_size=.18)
x_train = train.drop(columns=['TEY', 'CO', 'NOX'])
y_train = train.drop(columns=['AT','AH','AP','AFDP','GTEP','TIT','TAT','CDP', 'TEY', 'NOX'])
x_val = val.drop(columns=['TEY', 'CO', 'NOX'])
y_val = val.drop(columns=['AT','AH','AP','AFDP','GTEP','TIT','TAT','CDP', 'TEY', 'NOX'])
x_test = test.drop(columns=['TEY', 'CO', 'NOX'])
y_test = test.drop(columns=['AT','AH','AP','AFDP','GTEP','TIT','TAT','CDP', 'TEY', 'NOX'])
x_train = x_train.to_numpy()
y_train = y_train.to_numpy()
x_val = x_val.to_numpy()
y_val = y_val.to_numpy()
x_test = x_test.to_numpy()
y_test = y_test.to_numpy()

# set up prediction data
#data_2014 = pd.read_csv(r'Test/gt_2014.csv')
#x_predict = data_2014.drop(columns=['CO', 'NOX'])
#x_predict = x_predict.to_numpy()


# create layers and model
inputs = keras.Input(shape=(8,))
hidden1 = Dense(4, activation='relu')(inputs)
outputs = Dense(1, activation='linear')(hidden1)
model = keras.Model(inputs=inputs, outputs=outputs, name="GasTurbine_Model")

# summarize layers
print(model.summary())

# compile and fit model
model.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(),
    metrics=['mean_squared_error']
)
history = model.fit(x_train, y_train, batch_size=1, epochs=30, validation_data=(x_val,y_val))
plt.plot(history.history['loss'], label="loss")
#plt.plot(history.history['mean_squared_error'], label="mean_squared_error")
plt.plot(history.history['val_loss'], label="val_loss")
#plt.plot(history.history['val_mean_squared_error'], label="val_mean_squared_error")
plt.title('Model Results')
plt.legend(loc="upper left")
plt.xlabel('Epoch')
plt.show()

# evaluate model performance on test data
results = model.evaluate(x_test, y_test, batch_size=1)
print("test results: ", results)

# predict energy yield using prediction data on model
#prediction = model.predict(x_predict)
#print("predictions: ", prediction)
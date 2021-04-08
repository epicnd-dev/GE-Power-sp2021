import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import Activation, Dense
from sklearn.model_selection import train_test_split

# set up training and testing data
data_2011 = pd.read_csv(r'Train/gt_2011.csv')
train, test = train_test_split(data_2011, test_size=0.2)
x_training_data_2011 = train.drop(columns=['TEY', 'CO', 'NOX'])
y_training_data_2011 = train.drop(columns=['AT','AH','AP','AFDP','GTEP','TIT','TAT','CDP', 'CO', 'NOX'])
x_testing_data_2011 = test.drop(columns=['TEY', 'CO', 'NOX'])
y_testing_data_2011 = test.drop(columns=['AT','AH','AP','AFDP','GTEP','TIT','TAT','CDP', 'CO', 'NOX'])
x_training_data_2011 = x_training_data_2011.to_numpy()
y_training_data_2011 = y_training_data_2011.to_numpy()
x_testing_data_2011 = x_testing_data_2011.to_numpy()
y_testing_data_2011 = y_testing_data_2011.to_numpy()

# set up prediction data
prediction_data_2014 = pd.read_csv(r'Test/gt_2014.csv')
x_prediction_data_2014 = prediction_data_2014.drop(columns=['CO', 'NOX'])
x_prediction_data_2014 = x_prediction_data_2014.to_numpy()


# create layers and model
inputs = keras.Input(shape=(8,))
outputs = Dense(1, activation='linear')(inputs)
model = keras.Model(inputs=inputs, outputs=outputs, name="GasTurbine_Model")

# summarize layers
print(model.summary())

# compile and fit model
model.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(),
    metrics=['mean_squared_error']
)
history = model.fit(x_training_data_2011, y_training_data_2011, batch_size=64, epochs=50)

# evaluate model performance on test data
results = model.evaluate(x_testing_data_2011, y_testing_data_2011, batch_size=64)
print("test results: ", results)

# predict model performance on prediction data
prediction = model.predict(x_prediction_data_2014)
print("predictions: ", prediction)
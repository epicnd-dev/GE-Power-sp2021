import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import Activation, Dense
from sklearn.model_selection import train_test_split, KFold
from matplotlib import pyplot as plt

# set up data
data_2011 = pd.read_csv(r'../../Data/Train/gt_2011.csv')
data_2012 = pd.read_csv(r'../../Data/Train/gt_2012.csv')
frames = [data_2011, data_2012]
data = pd.concat(frames)
#train, test = train_test_split(data_2011, test_size=0.15)
#train, val = train_test_split(train, test_size=.18)
x_data = data.drop(columns=['TEY', 'CO', 'NOX'])
y_data = data.drop(columns=['AT','AH','AP','AFDP','GTEP','TIT','TAT','CDP', 'CO', 'NOX'])
#x_val = val.drop(columns=['TEY', 'CO', 'NOX'])
#y_val = val.drop(columns=['AT','AH','AP','AFDP','GTEP','TIT','TAT','CDP', 'CO', 'NOX'])
#x_test = test.drop(columns=['TEY', 'CO', 'NOX'])
#y_test = test.drop(columns=['AT','AH','AP','AFDP','GTEP','TIT','TAT','CDP', 'CO', 'NOX'])
x_data = x_data.to_numpy()
y_data = y_data.to_numpy()
#x_val = x_val.to_numpy()
#y_val = y_val.to_numpy()
#x_test = x_test.to_numpy()
#y_test = y_test.to_numpy()

# set up K-Fold Cross-Validation
no_splits = 5
kf = KFold(n_splits = no_splits, shuffle=True)
fold_no = 0
accuracy = [None] * no_splits
for train, test in kf.split(x_data):
    # set up training and validation data
    x_train, x_val = train_test_split(x_data[train], test_size=0.18)
    y_train, y_val = train_test_split(y_data[train], test_size=0.18)

    # create layers and model
    inputs = keras.Input(shape=(8,))
    hidden1 = Dense(6, activation='relu')(inputs)
    hidden2 = Dense(3, activation='relu')(hidden1)
    outputs = Dense(1, activation='linear')(hidden2)
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
    #plt.plot(history.history['loss'], label="loss")
    #plt.plot(history.history['mean_squared_error'], label="mean_squared_error")
    #plt.plot(history.history['val_loss'], label="val_loss")
    #plt.plot(history.history['val_mean_squared_error'], label="val_mean_squared_error")
    #plt.title('Model Results')
    #plt.legend(loc="upper left")
    #plt.xlabel('Epoch')
    #plt.show()

    # evaluate model performance on test data
    acc = model.evaluate(x_data[test], y_data[test], batch_size=1)
    accuracy[fold_no] = acc[0]
    fold_no = fold_no + 1

# calcuate and print model final results
print("Test Results: ", accuracy)
avg = np.mean(accuracy)
print("Average Result: ", avg)


# predict energy yield using prediction data on model
#prediction = model.predict(x_predict)
#print("predictions: ", prediction)
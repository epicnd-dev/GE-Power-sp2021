from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
import time

df1 = pd.read_csv('Data/Train/gt_2011.csv')
df2 = pd.read_csv('Data/Train/gt_2012.csv')
frames = [df1, df2]
df = pd.concat(frames)

# Configuration
# Can add or remove category names here to use as model inputs/outputs as needed
input_categories = ['GTEP', 'CDP', 'TIT', 'TAT']
output_categories = ['TEY']
# Name the model will be saved as
model_name = 'TEY_Model'
# Percentage of the data that will be used for training
# Percentage of data for validation = 1-training_split
training_data_split = 80
# Number of epochs the model will be trained with
epochs = 50

# Format the dataframe for Keras, based on the lists of categories above
input_df = pd.DataFrame(df, columns=input_categories)
output_df = pd.DataFrame(df, columns=output_categories)

data_len = len(input_df)
train_len = int(data_len*training_data_split/100)

# Split data into training/validation
input_data = input_df.iloc[:train_len,:]
output_data = output_df.iloc[:train_len,:]
input_validate = input_df.iloc[train_len:,:]
output_validate = output_df.iloc[train_len:,:]



# Model topology - add or remove layers as needed, play arount with activations, etc.
model = Sequential()

# Input layer and first layer of neurons
model.add(Dense(20, input_dim=len(input_df.columns), activation='relu'))

# Inner layers
model.add(Dense(10, activation='relu'))
model.add(Dense(20, activation='relu'))

# Output layer - number of outputs
model.add(Dense(len(output_df.columns), activation="linear"))

# Compile model - can play around with parameters here
model.compile(optimizer='adam', loss='mean_squared_error')

# Fit model with data (the actual training) - change # of epochs here
history = model.fit(x=input_data, y=output_data, epochs=epochs, validation_data=(input_validate, output_validate))



# Saving the model to files
# Create model name based on model_name + date/time (for now)
time_str = time.strftime("%Y%m%d-%H%M%S")
file_name = 'Models/Neural_Net/Models/' + model_name + '-' + time_str

# Save model training stats to a csv for future use
hist_df = pd.DataFrame(history.history) 
hist_csv_file = file_name + '.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

# Save the actual model to a file for future use
model.save(file_name)
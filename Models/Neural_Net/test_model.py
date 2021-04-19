import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error

# Desired inputs and outputs from the model
input_categories = ['GTEP', 'CDP', 'TIT', 'TAT']
output_categories = ["TEY"]

# Path[s] to data files that -do- have the output you're looking for
validation_paths = {
    2013 : "Data/Test/gt_2013.csv"
}

# Path[s] to data files that -do not- have the output you're looking for
generate_paths = {
    2014 : "Data/Test/gt_2014.csv",
    2015 : "Data/Test/gt_2015.csv"
}

# Load the model in
model_path = "Models/Neural_Net/Models/[NAME_OF_MODEL_HERE]"
#model_path = "Models/Neural_Net/Models/TEY_Model-20210331-195218"
model = load_model(model_path)

# Test how well the model predicts known values
for year in validation_paths.keys():
    df = pd.read_csv(validation_paths[year])

    # Format the dataframe for Keras, based on the lists of categories above
    input_df = pd.DataFrame(df, columns=input_categories)
    real_outputs = pd.DataFrame(df, columns=output_categories).to_numpy()

    predicted = model.predict(input_df)
    
    # Print the MSE
    print("MSE of predicted", year, "values vs expected values:", mean_squared_error(real_outputs, predicted))



# Predict unknown values
for year in generate_paths.keys():
    df = pd.read_csv(generate_paths[year])

    # Format the dataframe for Keras, based on the lists of categories above
    input_df = pd.DataFrame(df, columns=input_categories)

    predicted = model.predict(input_df)

    for i in range(len(output_categories)):
        # Save the predicted value to CSV
        file_path = "Models/Neural_Net/Predicted_Values/"
        file_name = str(year) + "_Predicted_" + output_categories[i] + "_[Insert_Model_Name_Here]"
        
        predicted = predicted[:,i]
        
        # Save the nunbers to a csv for future use
        pred_df = pd.DataFrame(predicted) 
        pred_csv_file = file_path + file_name + ".csv"
        with open(pred_csv_file, mode='w') as f:
            pred_df.to_csv(f)


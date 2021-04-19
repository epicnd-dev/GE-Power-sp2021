# WORK IN PROGRESS

import csv 
import numpy as np
from sklearn.linear_model import LinearRegression
from numpy import loadtxt
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold



df_train = pd.read_csv('Data/Train/gt_2011.csv', delimiter=',')
df_test = pd.read_csv('Data/Train/gt_2012.csv', delimiter=',')

inputs = ['AT','AP','AH','AFDP','GTEP','TIT','TAT']
outputs = ['CO', 'NOX']

input_df = pd.DataFrame(df_train, columns=inputs)
output_df = pd.DataFrame(df_train, columns=outputs)

X = input_df.values.tolist()
Y = output_df.values.tolist()



testin_df = pd.DataFrame(df_test, columns=inputs)
testout_df = pd.DataFrame(df_test, columns=outputs)

x = testin_df.values.tolist()
y = testout_df.values.tolist()

reg = LinearRegression().fit(X, Y)

print("The R^2 of the Model is: ", reg.score(X, Y))

pred2012 = reg.predict(x)

# np.savetxt("2012_Linear_Predictions.csv", pred2012, delimiter=" ")

plt.plot(pred2012)

plt.ylabel('TEY')
# plt.show()

print("The coefficients of each variable in the model: ", reg.coef_)


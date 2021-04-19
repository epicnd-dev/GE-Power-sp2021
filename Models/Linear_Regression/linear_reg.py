import csv 
import numpy as np
from sklearn.linear_model import LinearRegression
from numpy import loadtxt
import matplotlib.pyplot as plt
import pandas as pd

# use 2013 for testing
train_1 = pd.read_csv('Data/Train/gt_2011.csv', delimiter=',')
train_2 = pd.read_csv('Data/Train/gt_2012.csv', delimiter=',')
test_1 = pd.read_csv('Data/Test/gt_2013.csv', delimiter=',')
frames = [train_1, train_2]
df_train = pd.concat(frames)
print(len(df_train))

inputs = ['AT','AP','AH','AFDP','GTEP','TIT','TAT']
outputs = ['TEY']

input_df = pd.DataFrame(df_train, columns=inputs)
output_df = pd.DataFrame(df_train, columns=outputs)


X = input_df.values.tolist()
Y = output_df.values.tolist()


test_in = pd.DataFrame(test_1, columns=inputs)
test_out = pd.DataFrame(test_1, columns=outputs)

x = test_in.values.tolist()
y = test_out.values.tolist()

reg = LinearRegression().fit(X, Y)

print("The R^2 of the Model is: ", reg.score(x, y))

pred2013 = reg.predict(x)

np.savetxt("Models/Linear_Regression/2013_Linear_Predictions.csv", pred2013, delimiter=" ")

#plt.plot(pred2012)

#plt.ylabel('TEY')
# plt.show()

print("The coefficients of each variable in the model: ", reg.coef_)

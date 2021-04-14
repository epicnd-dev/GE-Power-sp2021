#%%

import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


df1 = pd.read_csv('../../Data/Train/gt_2011.csv')
df2 = pd.read_csv('../../Data/Train/gt_2012.csv')
frames = [df1, df2]
df = pd.concat(frames)

input_categories = ['GTEP', 'CDP', 'TIT', 'TAT']
output_categories = ['TEY']

input_df = pd.DataFrame(df, columns=input_categories)
output_df = pd.DataFrame(df, columns=output_categories)
input_d = input_df.values.tolist()
output_d = output_df.values.tolist()

k = 4
pca = PCA(n_components=k)
components = pca.fit_transform(input_d)
comp1 = list(components[:,0])
comp2 = list(components[:,1])
comp3 = list(components[:,2])
comp4 = list(components[:,3])

# Plot principal component 1 vs total energy yield
plt.scatter(comp1, output_d)
plt.show()

print("Explained variance per component:", pca.explained_variance_ratio_)

fname = "TEY_Princiapl_Components.csv"
components_df = pd.DataFrame(components, columns=range(1,len(components[0])+1))
components_df.to_csv(fname)
print("Saved components to", fname)
# %%

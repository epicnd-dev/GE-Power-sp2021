#!/usr/bin/env python3
from scipy import stats
import pandas as pd
import numpy as np
df = pd.read_csv('Train/gt_2011.csv')
z = np.abs(stats.zscore(df))
i = np.where(z>5)
print(list(zip(i[0],i[1])))
print("Number of outliers:",i[0].shape)
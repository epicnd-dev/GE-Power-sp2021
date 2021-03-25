#!/usr/bin/env python3
from scipy import stats
import pandas as pd
import numpy as np

df = pd.read_csv('Train/gt_2011.csv')

z = np.abs(stats.zscore(df))
i = np.where(z>7)
print(i[0].shape)
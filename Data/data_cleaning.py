#!/usr/bin/env python3
from scipy import stats
import pandas as pd
import numpy as np

def get_data(years, clean_data = False):
    
    # Dictionary linking year to file path
    file_names = {
        2011 : 'Data/Train/gt_2011.csv',
        2012 : 'Data/Train/gt_2012.csv',
        2013 : 'Data/Test/gt_2013.csv',
        2014 : 'Data/Test/gt_2014.csv',
        2015 : 'Data/Test/gt_2015.csv'
    }

    # Get all file paths requested
    files = []
    for year in years:
        files.append(file_names[year])

    # Read in all requested files into dataframes
    dfs = []
    for f in files:
        dfs.append(pd.read_csv(f))
    # Concat all dataframes into one df
    df = pd.concat(dfs)

    # Clean data if requested
    if(clean_data):
        z = np.abs(stats.zscore(df))
        i = np.where(z>5)
        print(list(zip(i[0],i[1])))
        print("Number of outliers:",i[0].shape)

    # Return the requested values
    return df


# Main method for testing (direct running)
if __name__ == '__main__':
    years = [2011, 2012]
    df = get_data(years)
    print(df)

    years = [2013,2014,2015]
    df2 = get_data(years, clean_data=True)
    print(df2)
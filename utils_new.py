import pandas as pd
import numpy as np


# Takes two dataframes and aligns them in the same frequency. Resamples so that each contains 
# equally many samples.
def align_and_resample(df1, df2, freq='15min', method='interpolate'):

    start = max(df1.index.min().ceil(freq), df2.index.min().ceil(freq))
    end = min(df1.index.max(), df2.index.max())

    # Check if dataframes overlap. 
    if start >= end:
        raise ValueError("No overlapping time range at given frequency.")

    df1_trimmed = df1.loc[start:end]
    df2_trimmed = df2.loc[start:end]

    df1_resampled = df1_trimmed.resample(freq).mean()
    df2_resampled = df2_trimmed.resample(freq).mean()

    if method == 'interpolate':
        df1_resampled = df1_resampled.interpolate()
        df2_resampled = df2_resampled.interpolate()
    elif method == 'ffill':
        df1_resampled = df1_resampled.ffill()
        df2_resampled = df2_resampled.ffill()
    elif method == 'bfill':
        df1_resampled = df1_resampled.bfill()
        df2_resampled = df2_resampled.bfill()

    return df1_resampled, df2_resampled

def calculate_mse(df1, df2):
    squared_diff=(df1-df2)**2
    mse=squared_diff.mean()
    return mse

def calculate_corr(df1,df2):
    corr=df1.corr(df2)
    return corr
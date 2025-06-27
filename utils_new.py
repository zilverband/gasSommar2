import pandas as pd
import numpy as np

# Takes two dataframes and aligns them in the same frequency. 
# Resamples so that each contains equally many samples.
def align_and_resample(df1, df2, freq='15min', method='interpolate'):

    df1=df1.tz_localize(None)
    df2=df2.tz_localize(None)

    start = max(df1.index.min(), df2.index.min())
    end = min(df1.index.max(), df2.index.max())

    # Check if dataframes overlap. 
    if start >= end:
        raise ValueError("No overlapping time range at given frequency.")

    df1_trimmed = df1.loc[(df1.index >= start) & (df1.index <= end)]
    df2_trimmed = df2.loc[(df2.index >= start) & (df2.index <= end)]

    df1_resampled = df1_trimmed.resample(freq).mean()
    df2_resampled = df2_trimmed.resample(freq).mean()

    # Interpolate to replace the missing value.
    if method == 'interpolate':
        df1_resampled = df1_resampled.interpolate()
        df2_resampled = df2_resampled.interpolate()
    # Replace value with the latest valid observation.
    elif method == 'ffill':
        df1_resampled = df1_resampled.ffill()
        df2_resampled = df2_resampled.ffill()
    # Replace value with the next valid observation.
    elif method == 'bfill':
        df1_resampled = df1_resampled.bfill()
        df2_resampled = df2_resampled.bfill()

    df1_resampled, df2_resampled = df1_resampled.align(df2_resampled, join='inner')
    
    mask = df1_resampled.notna().all(axis=1) & df2_resampled.notna().all(axis=1)
    df1_final = df1_resampled.loc[mask]
    df2_final = df2_resampled.loc[mask]

    return df1_final, df2_final

# Calculate mean squared error.
def calculate_mse(df1, df2):
    squared_diff=(df1-df2)**2
    mse=squared_diff.mean()
    return mse

# Calculate correlation.
def calculate_corr(df1,df2):
    corr=df1.corr(df2)
    return corr
import pandas as pd
import numpy as np
import scipy
from scipy.stats import median_abs_deviation

# Takes two dataframes and aligns them in the same frequency. 
# Resamples so that each dataframe contains equally many samples.
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

    return df1_resampled, df2_resampled

# Calculate mean squared error.
def calculate_mse(df1, df2):
    squared_diff=(df1-df2)**2
    mse=squared_diff.mean()
    return mse

# Calculate correlation.
def calculate_corr(df1,df2):
    corr=df1.corr(df2)
    return corr

# LP-filtering
def lp_filter(df, T, cutoff):
    dates = df.index
    signal = df.to_numpy().flatten()
    fs = 1/(T*60)
    Wn = cutoff/(2*np.pi)/(fs/2)
    b, a = scipy.signal.butter(7,Wn,btype='low')
    yf = scipy.signal.filtfilt(b, a, signal)

    # For plotting
    """N = len(signal)
    n = np.arange(N)
    Y = np.fft.fft(signal)
    plt.figure(0)
    plt.stem(2*np.pi*fs*n/N,np.abs(Y),basefmt="")
    plt.xlim(0,0.002)
    plt.title("Original FFT")
    YF = np.fft.fft(yf)"""

    return dates, yf

def hampel_filter(data, window_size=5, n_sigmas=3):
    data = data.copy()
    N = len(data)
    k = window_size
    for i in range(k,N-k):
        window = data[(i-k):(i+k+1)]
        median = np.median(window)
        mad = median_abs_deviation(window, scale='normal')
        if np.abs(data[i]-median)>(n_sigmas*mad):
            data[i]=median

    return data
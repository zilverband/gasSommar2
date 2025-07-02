import pandas as pd
import numpy as np
import scipy
from scipy.stats import median_abs_deviation
import functions.utils as u

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

    return pd.DataFrame(yf, dates)

def hampel_filter(data,n_sigmas=2):
    mu = data.mean()
    sigma = data.std()

    return data[np.abs(data-mu) < n_sigmas*sigma]

# Takes temp or humidity data and detects quick changes 
"""def c_derivative(data):
    dates = data.index
    dt = data.index.to_series().diff()

    dt_int = dt.to_numpy().astype('float')
    data_array = data.to_numpy()

    data_array_int = []
    dt_int_new = []
    new_dates = []

    for i, value in enumerate(data_array):
        if value != "null":
            data_array_int.append(float(value))
            dt_int_new.append(dt_int[i])
            new_dates.append(dates.to_numpy()[i])

    dy_int = np.diff(data_array_int)

    diff = dy_int / dt_int_new[:-1]

    return new_dates[:-1], diff"""

def calc_derivative(df):
    dt = df.index.to_series().diff().dt.total_seconds()
    dy = df.diff()
    diff = dy/dt
    return diff

def warning_fcn(array, limit):
    warn_array = np.full(len(array), False)
    counter = 0
    for temp in array:
        if (np.abs(temp) > limit):
            warn_array[counter] = True
        counter = counter + 1
    return warn_array
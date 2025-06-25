"""
data_processing.py

This file contains functions related to data loading, cleaning, and basic manipulation. 
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

from . import utils as ut


def before(data, end_time):
    """
    Return data before specified end time.
    """
    return data[data.index <= end_time]


def after(data, start_time):
    """
    Return data after specified start time.
    """
    return data[data.index >= start_time]


def between(data, start_time, end_time):
    """
    Return data between specified times.
    """
    return after(before(data, end_time), start_time)


def mean_and_std(data):
    """
    Return the mean and standard deviation of the data.
    """
    return data.mean(), data.std()


def normalize(df, scaler=None):
    """
    Normalize the data. Assumes the scaler to be fitted.
    """
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(df)

    normalized_data = pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)

    return normalized_data


def match(df1, df2, start_time=None, end_time=None):
    """
    Match two dataframes to each other.

    Assumes that the dataframes already partially overlaps.
    Will remove NaN values.
    """

    # Set start and end times
    if not start_time:
        start_time = max(df1.index[0], df2.index[0])
    if not end_time:
        end_time = min(df1.index[-1], df2.index[-1])

    # Only keep data inside time range
    df1 = between(df1, start_time, end_time)
    df2 = between(df2, start_time, end_time)

    # Find and remove differing indexes
    diff1 = df1.index.difference(df2.index)
    diff2 = df2.index.difference(df1.index)
    df1.drop(diff1, inplace=True)
    df2.drop(diff2, inplace=True)

    return df1, df2


def append_report_data(sensor_data, report_data):
    """
    Append report data to API data.
    """
    sensor_data, report_data = match(sensor_data, report_data)
    data = pd.concat([sensor_data, report_data], axis=1)
    return data


def aggregate_duplicates(data):
    """
    Take average of duplicate rows and remove all but one.
    """
    # data = data.groupby(data.index).agg({col: "mean" for col in data.columns})
    data = data.groupby(data.index).agg({col: "first" if isinstance(data[col].iloc[0], str) else "mean" for col in data.columns})
    return data


def match_to_ref(data, ref_data, shift=0):
    """
    Matches data to a reference time window, averaging values within a specified time shift.
    """
    # Matches ugm3 data to reference time [OBS: INEFFECTIVE!!! - use resample_and_match instead]
    window = int(ut.minutes_between(ref_data.index[0], ref_data.index[1]))

    matched_data = {feature : [] for feature in data.columns}
    times = []

    shift_delta = pd.Timedelta(minutes=shift)
    time_delta = pd.Timedelta(minutes=window)

    for time in ref_data.index:
        times.append(time)
        for feature in data.columns:
            quarter_data = between(data, time+shift_delta-time_delta, time+shift_delta+time_delta)
            matched_data[feature].append(quarter_data[feature].mean())

    return pd.DataFrame(matched_data, index=times)


def resample_and_match(df1, df2, freq_str, start_time=None, end_time=None):
    """
    Resample the data frames and match them.

    Returns resampled versions of the dataframes where both have matching indexes.
    """
    freq = ut.freqstring_to_int(freq_str)
    avg_time_diffs = (ut.avg_time_diff(df1), ut.avg_time_diff(df2))

    # Resample and drop NaN-values
    df1_resampled = df1.copy().resample(freq_str).agg({col: "first" if isinstance(df1[col].iloc[0], str) else "mean" for col in df1.columns})
    df2_resampled = df2.copy().resample(freq_str).agg({col: "first" if isinstance(df2[col].iloc[0], str) else "mean" for col in df2.columns})

    if freq < avg_time_diffs[0]:
        print("Mvg avg df1")
        df1_resampled = df1_resampled.interpolate("linear")
    if freq < avg_time_diffs[1]:
        print("Mvg avg df2")
        df2_resampled = df2_resampled.interpolate("linear")

    df1_resampled.dropna(inplace=True)
    df2_resampled.dropna(inplace=True)

    # Filter according to times
    if not start_time:
        start_time = max(df1_resampled.index[0], df2_resampled.index[0])
    if not end_time:
        end_time = min(df1_resampled.index[-1], df2_resampled.index[-1])
    df1_resampled = between(df1_resampled, start_time, end_time)
    df2_resampled = between(df2_resampled, start_time, end_time)

    return match(df1_resampled, df2_resampled, start_time, end_time)


def resample(data, factor):
    """
    Resamples data by increasing the number of points based on 
    a specified factor using linear interpolation.
    """
    original_length = len(data)
    new_length = int(original_length * factor)
    new_indices = np.linspace(0, original_length - 1, new_length)
    new_data = pd.DataFrame()

    for feature in data.columns:
        # Create the interpolation function for each column and interpolate the data
        interpolation = interp1d(np.arange(original_length), data[feature], kind="linear")
        new_data[feature] = interpolation(new_indices)

    start_time = data.index[0]
    end_time = data.index[-1]
    new_data.index = pd.date_range(start=start_time, end=end_time, periods=new_length)

    return new_data


def smooth_data(data, do_smoothing=True, alg="Gauss", window_size=6, poly_order=3, sigma=3,
                      do_peak_threshold=True, std_threshold=3):
    """
    Smooths data using specified algorithms (Savitzky-Golay, moving average, Gaussian)
    and applies optional peak thresholding.
    """

    for feature in data.columns:
        # Ensure that the std is not calculated for a string.
        if type(data[feature].iloc[0]) == str:
            continue

        # Apply a threshold of std_threshold standard deviations from the mean on the data.
        if do_peak_threshold:
            std = np.sqrt(np.var(data[feature], ddof=1))
            mean = np.mean(data[feature])
            data[feature] = data[feature].apply(lambda x: x
                if abs(x) < mean + std_threshold*std else mean + std_threshold*std)

        # Data smoothing
        if not do_smoothing: return data
        if alg.lower().strip() == "savgol":
            data[f"smooth_{feature}"] = savgol_filter(data[feature], window_size, poly_order)
        elif alg.lower().strip() == "moving":
            data[f"smooth_{feature}"] = np.convolve(data[feature],
                                                    np.ones(window_size)/window_size, mode='same')
        elif alg.lower().strip() == "gauss":
            data[f"smooth_{feature}"] = gaussian_filter1d(data[feature], sigma)

    return data


def get_filter_mask(data, feature, min, max):
    """
    Generates a boolean mask for filtering data based on a feature's value falling within a specified range.
    """
    mask = (data[feature] >= min) & (data[feature] < max)
    return list(mask)


def temp_correct_baseline(data, color="turquoise"):
    """
    Corrects temperature baseline data using polynomial coefficients 
    based on a specified color.
    """
    temp = data["temp"]
    coeffs = {
        "middle dark blue" : ut.generate_coeffs([15, 13, 10, 8, 6, 1, -9, -13, 7]),
        "dark blue" : ut.generate_coeffs([1.230769231, 1.076923077, 0.8461538462, 1, 0.9230769231, 1, 1.230769231, 2.538461538, 8.615384615]),
        "turquoise" : ut.generate_coeffs([1.444444444, 1.333333333, 1.222222222, 1, 0.8888888889, 1, 0.7777777778, 1.666666667, 8.222222222]),
        "pink" : ut.generate_coeffs([1.5, 1.833333333, 2, 1.833333333, 1.333333333, 1, -0.5, -0.8333333333, 2.5]),
        "red" : ut.generate_coeffs([2.777777778, 2.333333333, 2, 1.555555556, 1.333333333, 1, 0, -0.5555555556, 3.444444444]),
        "dark turquoise" : ut.generate_coeffs([12, 11, 9, 7, 4, 1, -1, 20, 45]),
        "brown" : ut.generate_coeffs([7, 5.5, 4.5, 3, 2, 1, -2, -1, 24.5]),
        "purple" : ut.generate_coeffs([1.625, 1.375, 1.125, 1.125, 1, 1, 0.625, 1.875, 9.25]),
        "eval complex" : ut.generate_coeffs([201429.87189128099, 33178.08162744186, 3044.6751656845445, 74.67611355342514, 1.0595986999674807, 1.0040090733663334, 0.9841953785372795, 10.788126057225682, 984.8380118688407]),
        #"eval simple" : ut.generate_coeffs([21938.902412852574, 4622.154000464096, 644.2977854310998, 43.3431332966637, 1.414798606283405, 1.0081301976066825, 0.9750587651982769, -0.00365143025535275, -35.271866049144954])}
        "eval simple" : ut.generate_coeffs([-15899.365849198206, -2855.129690327841, -310.9912303727725,-12.15834366215864, 0.9923226186535068, 1.0106202990590027, 0.9813976707815222, 0.55286601841485, -42.89870290414888]),
        'strandv deg2': [-1.01486280e-04,  2.27462073e-03,  1.00531205e+00],
        'accumulated deg2': [-5.64936750e-05, -2.41468742e-04,  1.03424007e+00]
        }

    corrected_baseline = [ut.evaluate_polynomial(coeffs[color], t) for t in temp]

    data["temp_corr"] = corrected_baseline
    data["WEoc"] = data["WEo"]*data["temp_corr"]
    data["WEuc"] = data["WEu"]/data["temp_corr"]
    data["AEoc"] = data["AEo"]*data["temp_corr"]
    data["AEuc"] = data["AEu"]/data["temp_corr"]
    data["WET"] = data["WEuc"] - data["WEo"]


def get_ref_WEc(ref, data):
    """
    Calculates the reference weighted equivalent concentration (WEc)
    based on pressure, temperature, and sensor gain.
    """
    M = 46.0055
    R = 8.314
    A = M*data["pressure"]*100/(R*(data["temp"]+273.15)*data["we_sensor"]*data["gain"])
    return ref["ugm3"] / A


def remove_outliers(df, threshold=3):
    """
    Removes outliers from a DataFrame based on their deviation from
    the mean by a threshold.
    """

    numeric_df = df.select_dtypes(include=[np.number])

    scaler = StandardScaler()
    scaler.fit(numeric_df)
    z_scores = normalize(numeric_df, scaler)

    outliers = (np.abs(z_scores) > threshold).any(axis=1)

    outliers.index = df.index
    # Identify outliers

    # Filter out the outliers
    cleaned_df = df[~outliers]

    return cleaned_df.copy()


def create_sequences(data, n_steps):
    """
    Creates sequences of data with a specified number of steps
    for time series analysis.
    """
    sequences = []
    for i in range(len(data) - n_steps + 1):
        sequences.append(data[i:i + n_steps])
    return np.array(sequences)

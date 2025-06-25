"""
utils.py

Utility functions for file handling, data processing, and other common tasks.
"""

import numpy as np

def minutes_between(time1, time2):
    """
    Calculate the number of minutes between two datetime objects.
    """
    return (time2 - time1).total_seconds()/60


def freqstring_to_int(freq):
    """
    Convert frequence string to minutes.
    """
    int_part = ""
    str_part = ""

    for c in freq:
        if c.isdigit():
            int_part += c
        else:
            str_part += c

    return int(int_part) * (60 if str_part!="min" else 1)


def avg_time_diff(data):
    """
    Calculate average time difference.
    """
    return round(data.index.to_series().diff().median().total_seconds()/60)


def generate_coeffs(factors):
    """
    Generates polynomial coefficients to fit the given temperature compensation factors.
    """
    # Static temperature range base on the AlphaSense Zero background
    # current temperature compensation factors
    temperatures = np.array([-30, -20, -10, 0, 10, 20, 30, 40, 50])
    degree = len(factors) - 1
    coefficients = np.polyfit(temperatures, factors, degree)
    return coefficients


def evaluate_polynomial(coefficients, x):
    """
    Evaluates a polynomial at a given point using the provided coefficients.
    """
    result = 0
    for index, coeff in enumerate(coefficients):
        power = len(coefficients) - 1 - index
        result += coeff * (x ** power)

    return result


def get_derivative(df, feature, sample_time="hour"):
    """
    Calculates the derivative of a specified feature in a DataFrame, 
    scaled by the specified sample time (hour or minute).
    """
    der = df[feature].diff() / df.index.to_series().diff().dt.total_seconds()
    if sample_time.lower().strip() == "hour":
        der *= 60*60
    elif (sample_time.lower().strip() == "min") or (sample_time.lower().strip() == "minute"):
        der *= 60
    return der

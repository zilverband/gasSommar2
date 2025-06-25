"""
data_io.py

This file contains functions related to loading and saving data.
"""

import os
import pandas as pd
from . import data_processing as dp

# All features from the API-data to save in DataFrame, except response features
SENSOR_FEATURES = {
    "sensor_type" : "sensor_type",
    "weu" : "WEu",
    "wee" : "WEe",
    "weo" : "WEo",
    "aeu" : "AEu",
    "aee" : "AEe",
    "aeo" : "AEo",
    "temperature" : "temp",
    "gain" : "gain",
    "we_sensor" : "we_sensor",
    "pressure" : "pressure",
    "vref_voltage" : "vref_voltage",
    "offset_constant" : "offset_constant",
    "offset_factor" : "offset_factor"
}

# Specifies what columns from reference data to save in DataFrame, and what
# the new columns should be called.
REF_COLUMNS =  {
                "strandv" : {"Hamngatan 10 (ug/m3)" : "ugm3",
                            "Urban bakgrund, Norrkoping (ug/m3)" : "urban"},
                "strandv2" : {"Hamngatan 10 (ug/m3)" : "ugm3",
                            "Urban bakgrund, Norrkoping (ug/m3)" : "urban"},
                "torkel" : {"NO2" : "ugm3", 
                            "NO" : "ugm3_NO",
                            "CO2" : "ugm3_CO2",
                            "O3" : "ugm3_O3",
                            "temp" : "temp"},
                "svea" : {"NO2" : "ugm3",
                          "NO" : "ugm3_NO",
                          "CO" : "ugm3_CO"},
                "italien" : {"NO2" : "ugm3",
                             "NO" : "ugm3_NO",
                             "NOX" : "ugm3_NOX",
                             "SO2": "ugm3_SO2"}
                }


def load_api_data(filename):
    """
    Loads API data from single json file.
    """
    raw_data = pd.read_json(filename)
    response_features = ["ppb", "ugm3", "raw_ppb", "raw_ugm3", "WEc", "correctionFactor"]
    data_dict = {}

    # Extract columns according to SENSOR_FEATURES
    for feature in raw_data.columns:
        if feature in SENSOR_FEATURES:
            if feature != "sensor_type":
                data_dict[SENSOR_FEATURES[feature]] = pd.to_numeric(raw_data[feature])
            else:
                data_dict[SENSOR_FEATURES[feature]] = raw_data[feature]

    # Extract columns according to response features
    for feature in response_features:
        data_dict[feature] = pd.to_numeric(
            raw_data["response"].apply(lambda x: x[feature] if feature in x else None))

    # Turn into data frame object
    data = pd.DataFrame(data_dict)
    data.index = pd.to_datetime(raw_data["received_date"])
    data.dropna(inplace=True)

    return data


def load_gas_data(sensor_name, gas, add_suffix=False):
    """
    Loads data for the specified gas.
    """
    path = f"data/{sensor_name}/sensor/" # Find path
    add_on = "1" if sensor_name == "strandv" else "" # Potential add on

    # Create list of filenames
    filenames = [path + file for file in os.listdir(path) if
                  file.startswith(f"{sensor_name}{add_on}_{gas}")]

    # Load first file
    data = load_api_data(filenames[0])

    # Load new files and add to existing data frame
    for filename in filenames[1:]:
        new_data = load_api_data(filename)
        same = new_data.index.intersection(data.index)
        new_data.drop(same, inplace=True)
        data = pd.concat([data, new_data])

    data.sort_index(inplace=True)

    # Add suffixes to column names
    if add_suffix:
        data.columns = [f"{col}_{gas}" for col in data.columns]

    return data


def combine_gas_data(sensor_name, gasses):
    """
    Combines multiple gas datas into one data frame.
    """
    # Create list of data frames, each containing data from one gas
    dfs = [dp.aggregate_duplicates(load_gas_data(sensor_name, gas, add_suffix=(len(gasses) != 1))) for gas in gasses]

    # Find intersecting indexes
    same_idx = dfs[0].index
    for df in dfs[1:]:
        same_idx = same_idx.intersection(df.index)

    # Merge all data frames into one
    data = pd.concat([df.loc[same_idx] for df in dfs], axis=1)

    return data


def read_csv_data(filename):
    """
    Read a CSV file and return its contents as a DataFrame, 
    using the first row as column headers.
    """
    with open(filename) as f:
        lines = [line.rstrip().split(";") for line in f.readlines()]

    # Find relevant starting index
    for i in range(len(lines)):
        if lines[i][0] == "Datum":
            break

    ref_data = pd.DataFrame(lines[i+1:])
    ref_data.columns = lines[i]

    return ref_data


def load_ref_data(filename):
    """
    Load reference data for specified file.
    """
    station_name = filename.split("/")[1]
    ref_data = read_csv_data(filename)

    # Compensate for different date conventions and add time as index.
    add_on = ["" if date.startswith("2024") else "20" for date in ref_data["Datum"]]
    ref_data.index = pd.to_datetime(add_on+ref_data["Datum"]+" "+ref_data["Kl"])
    ref_data.index = ref_data.index.tz_localize("Europe/Berlin") # Timezone CEST

    columns_to_keep = REF_COLUMNS[station_name]

    # Add columns with right names
    for col in list(ref_data.columns):
        for key, new_name in columns_to_keep.items():
            if key == col:
                ref_data[new_name] = pd.to_numeric(ref_data[col])

    ref_data = ref_data[columns_to_keep.values()]
    ref_data.dropna(inplace=True)

    return ref_data


def load_report_data(filename):
    """
    Load data from report file.
    """
    # Open file
    with open(filename) as f:
        lines = [line.rstrip().split(";") for line in f.readlines()]

    # Extract data and column names
    data = pd.DataFrame(lines[1:])
    data.columns = lines[0]

    # Convert to appropriate format
    data["temp_ex"] = pd.to_numeric(data["temp_external"])
    data["humidity_ex"] = pd.to_numeric(data["humidity_external"])
    data["datetime"] = pd.to_datetime(data["timestamp"])

    # Extract relevant columns and set index
    data = data[["datetime", "humidity_ex", "temp_ex"]]
    data.set_index("datetime", inplace=True)
    data.index = data.index.tz_localize("Europe/Stockholm")
    data.dropna(inplace=True)

    return data


def get_filename(sensor_name, subfolder, keyword):
    """
    Return correct filename.
    """
    path = f"data/{sensor_name}/{subfolder}/"
    add_on = "1" if sensor_name == "strandv" else ""
    file = [path + file for file in os.listdir(path)
            if file.startswith(f"{sensor_name}{add_on}_{keyword}")][0]
    return file


def load_data(sensor_name, gasses = ["NO2"]):
    """
    Load sensor and reference data.
    """
    # Must pass gasses as a list!
    sensor_data = combine_gas_data(sensor_name, gasses)
    report_data = dp.aggregate_duplicates(load_report_data(get_filename(sensor_name, "sensor", "report")))
    sensor_data = dp.append_report_data(sensor_data, report_data)
    sensor_data = dp.aggregate_duplicates(sensor_data)
    ref_data = load_ref_data(get_filename(sensor_name, "ref", "ref"))

    return sensor_data, ref_data

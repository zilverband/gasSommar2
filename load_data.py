import json
import pandas as pd
import matplotlib.pyplot as plt
import datetime

#Namnen på olika mätningar från de olika filerna.
NAME_DICT = {
    'pm1_0' : 'PM1',
    'pm2_5' : 'PM2.5',
    'pm10_0' : 'PM10',
    'pc1_0' : 'PC1',
    'pc2_5' : 'PC2.5',
    'pc10_0' : 'PC10',
    'no2_gas' : 'NO2',
    'o3_gas' : 'O3',
    'co_gas' : 'CO',
    'humidity_external' : 'rel_hum',
    'co2_gas' : 'CO2',
    'no_gas' : 'NO'
}

#Takes json data from olb and returns a panda dataframe with data, and a panda series with data descriptions.
def load_slb_data(file):
    data_json = json.load( open(file,'r'))
    data = pd.DataFrame.from_records(data_json["data"]).transpose()
    data.index = pd.to_datetime(data.index)

    desc = data.columns
    data.columns = desc.str.split().str[0]
    desc = pd.Series(desc,index=data.columns)
    return data, desc

#Takes csv files and returns a panda dataframe with data and a panda series with data descriptions
def load_sensor_data(file):
    data = pd.read_csv(file)
    data.index = pd.to_datetime(data['timestamp'])
    
    #Jag gissade lite vad som är intressant det finns mer
    data = data[['pm1_0 (µg/m³)','pm2_5 (µg/m³)','pm10_0 (µg/m³)',
                'pc1_0 (#/cm³)', 'pc2_5 (#/cm³)','pc10_0 (#/cm³)','pressure_aq (hPa)',
                'no2_gas (ppb)','o3_gas (ppb)','co_gas (ppb)','temp_external (°C)',
                'humidity_external (% RH)','co2_gas (ppm)','no_gas (ppb)']]

    for sensor_name,ref_name in NAME_DICT.items():
        data.columns = data.columns.str.replace(sensor_name,ref_name)
    desc = data.columns
    data.columns = desc.str.split().str[0]
    desc = pd.Series(desc,index=data.columns)
    data = data.dropna()
    
    return data, desc
import requests
import pandas as pd
from datetime import datetime, timedelta
#'https://console.sensorbee.com/api/report?installationid=LKPG_Strandv_2&from=2025-03-01&to=2025-03-14'


URL = 'https://console.sensorbee.com/api'
URL_REF = 'https://open.slb.nu/api-dev/v0.1b/'

ACTIONS = {
    "report" : "/report",
    "raw" : "/gas-raw-values"
}

HEADERS = {
    "apiKey" : "ac32fe3ed21c60686e0d399a1ad506d2"
}

GAS_STYPE = {
    "NO2" : "NO2-B43F",
    "NO" : "NO-B4",
    "CO" : "CO-B4",
    "O3" : "OX-B431" 
}

STATIONS = {
    "svea" : "api-sveav59-kvartar",
    "torkel" : "api-torkel-kvartar",
    "linkoping" : "api-linkoping-kvartar"
}


#NOT RECOMMENDED
def get_raw_gas_data(end_name,gasses,start,end,file=None):
    #Define which information to get from API
    info = {
        "endpoint_name" : end_name,
        "from" : start,
        "to" : end,
        "sensor_type" : ""
    }
    data_list = []

    for gas in gasses:
        new = get_one_gas(info,gas)
        new = new[~new.index.duplicated()]
        data_list.append(new) 
    
    data = combine_data(data_list)
    if file is not None:
        data.to_csv(file + ".csv")
        print("data saved to: " + file + ".csv")
    return data

def combine_data(data_list):
    data = pd.concat(data_list,axis=1)
    data = data.T.groupby(by=data.columns).mean().T
    return data

def get_report_data_long(dataset,install_id,features={"temp_external": "temp", "humidity_external":"hum"},return_raw=False):
    start = dataset.index[0]
    end = dataset.index[-1]
    delta = timedelta(weeks=4)
    data = None

    while start + delta < end:
        if data is None:
            data = get_report_data(start,start+delta,install_id,features=features)
        else:
            data = pd.concat([data, get_report_data(start,start+delta,install_id,features=features)])
        start += delta
    
    #Gets the last bit of data

    if data is None:
        data = get_report_data(start,end,install_id,features=features)
    else:
        data = pd.concat([data, get_report_data(start,end,install_id,features=features)])

    if return_raw:
        return data
    
    dataset = pd.concat([dataset,data],axis=1)
    dataset = dataset.dropna(thresh=(len(features.keys())+1))
    return dataset


def get_report_data(start,end,install_id,dataset=None,features={"temp_external": "temp", "humidity_external":"hum"}):
    '''Takes report data and appends them to dataset.'''
    action = "report"
    info = {"installationid" : install_id,
            "from" : start,
            "to" : end,
            }
    
    response = requests.get(URL + ACTIONS[action],headers=HEADERS, params=info)

    raw_data = response.json()
    #return raw_data
    raw_data_df = pd.DataFrame(raw_data)
    parsed_data = {}
    
    for feature in raw_data_df.columns:
        if feature in features:
            parsed_data[features[feature]] = raw_data_df[feature].to_list()
    
    data = pd.DataFrame(parsed_data,index=pd.to_datetime(raw_data_df['timestamp'], unit='ms'))
    data.index.name = None
    data.index = data.index.tz_localize(None)
    data = data[~data.index.duplicated()]
    
    print(dataset is None)
    if dataset is None:
        return data
    else:
        dataset = pd.concat([dataset,data],axis=1)
        dataset = dataset.dropna(thresh=(len(features.keys())+1))
        return dataset

def get_ref_data(station,start,end,file):
    url = URL_REF + STATIONS[station]
    info = {
        "from_date" : start,
        "to_date" : end
    }

    response = requests.get(url,params=info)
    with open(file + ".json", "w") as f:
        f.write(response.text)
        print("data saved to: " + file + ".json")
    
def load_csv(file):
    data = pd.read_csv(file,index_col=0)
    data.index = pd.to_datetime(data.index)
    return data

def get_one_gas_long(start,end,end_name,gas,file=None):
    "Gets API data by 4 weeks intervals, stabler and faster"
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    delta = timedelta(weeks=4)
    data = None
    while start + delta < end:
        info = {"endpoint_name" : end_name,
        "from" : start.strftime('%Y-%m-%d'),
        "to" : (start+delta).strftime('%Y-%m-%d'),
        }
        if data is None:
            data = get_one_gas(info,gas)
        else:
            data = pd.concat([data, get_one_gas(info,gas)])
        start += delta
    
    #Gets the last bit of data
    info = {"endpoint_name" : end_name,
        "from" : start.strftime('%Y-%m-%d'),
        "to" : (end).strftime('%Y-%m-%d'),
        }
    if data is None:
        data = get_one_gas(info,gas)
    else:
        data = pd.concat([data, get_one_gas(info,gas)])


    if file is not None:
        data.to_csv(file + ".csv")
        print("data saved to: " + file + ".csv")

    return data



#Returns raw and calibrated gas data and offset
def get_one_gas(info,gas,file=None):
    '''
    Gets raw and calibrated gas data and offset from the API

    :param info: dictionary with keys {"start" (YYYY-MM-DD), "end" (YYYY-MM-DD), "endpoint_name"}
    :gas: string with the desired gas, ("NO","NO2","CO","O3")
    :return: pandas dataframe
    '''
    action = "raw"
    info["sensor_type"] = GAS_STYPE[gas]
    
    #Gets data from API
    response = requests.get(URL + ACTIONS[action],headers=HEADERS, params=info)
    
    #Converts the data into dataframes
    raw_data = response.json()
    raw_data_df = pd.DataFrame(raw_data).dropna(subset='response')
    response_df = pd.DataFrame(raw_data_df['response'].to_list())
    parsed_data = {}
    
    #interesting features
    sensor_features = {
        "offset_constant" : 'offset'
    }
    response_features = {
        "raw_ppb" : "(raw)",
    }

    #Collects desired features from data
    for feature in raw_data_df.columns:
        if feature in sensor_features:
            parsed_data[sensor_features[feature]] = raw_data_df[feature].to_list()

    #Gas data is in subset "response"
    for feature in response_df.columns:
        if feature in response_features:
            parsed_data[gas] = response_df[feature].to_list()
    
    #Assembles data into a dataframe
    data = pd.DataFrame(parsed_data,index=pd.to_datetime(raw_data_df['received_date'])) 
    data.index.name = None
    data.index = data.index.tz_localize(None)
    data.dropna()
    data = data[~data.index.duplicated()]

    #Calculates the uncalibrated data
    data[gas+"_raw"] = data[gas] + data["offset"]

    #Saves file
    if file is not None:
        data.to_csv(file + ".csv")
        print("data saved to: " + file + ".csv")
    return data
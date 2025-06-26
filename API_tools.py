import requests
import pandas as pd
#'https://console.sensorbee.com/api/report?installationid=LKPG_Strandv_2&from=2025-03-01&to=2025-03-14'


URL = 'https://console.sensorbee.com/api'

ACTIONS = {
    "report" : "/report",
    "raw" : "/gas-raw-values"
}

KEY = "ac32fe3ed21c60686e0d399a1ad506d2"

GAS_STYPE = {
    "NO2" : "NO2-B43F",
    "NO" : "NO-B4",
    "CO" : "CO-B4",
    "O3" : "OX-B431" 
}


#Returns a data frame with raw gas data for one gas
def get_raw_gas_data(end_name,gas,start,end):
    #Define which information to get from API
    info = {
        "endpoint_name" : end_name,
        "from" : start,
        "to" : end,
        "sensor_type" : GAS_STYPE[gas]
    }

    headers = {
        "apiKey" : KEY
    }

    action = "raw"

    #Gets data from API
    response = requests.get(URL + ACTIONS[action],headers=headers, params=info)
    

    #Converts dictionaries into better format (dictionary containing lists, not vice versa)
    raw_data = response.json()
    raw_data_df = pd.DataFrame(raw_data).dropna(subset='response')
    try:
        response_df = pd.DataFrame(raw_data_df['response'].to_list())
    except:
        print("FAIL")
        return raw_data_df
    print("SUCCESS")
    
    parsed_data = {}
    
    #interesting features
    sensor_features = {
        "temperature" : "temp"
    }
    response_features = {
        "ugm3" : "",
        "raw_ugm3" : "(raw)",
        "default_raw_ugm3" : " (default raw)"
    }

    #Collects desired features from data
    for feature in raw_data_df.columns:
        if feature in sensor_features:
            parsed_data[sensor_features[feature]] = raw_data_df[feature].to_list()

    #Gas data is in subset "response"

    for feature in response_df.columns:
        if feature in response_features:
            parsed_data[gas + "" + response_features[feature]] = response_df[feature].to_list()

    data = pd.DataFrame(parsed_data,pd.to_datetime(raw_data_df['received_date']))
    data.index.name = None
    return data
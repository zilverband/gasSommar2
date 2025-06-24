import json
import pandas as pd
import matplotlib.pyplot as plt
import datetime

#Takes json data from olb and returns a panda dataframe with data, and a panda series with data descriptions.
def load_slb_data(file):
    data_json = json.load( open(file,'r'))
    data = pd.DataFrame.from_records(data_json["data"]).transpose()
    data.index = pd.to_datetime(data.index)

    desc = data.columns
    data.columns = desc.str.split().str[0]
    desc = pd.Series(desc,index=data.columns)
    return data, desc
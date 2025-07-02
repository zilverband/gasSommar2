import API_tools as api
import pandas as pd

FILEPATH_REF = "data/refdata/"
FILEPATH_SENS = "data/sensordata/"
SENSORS = {
    "linkoping1" : "sb_8885",
    "linkoping2" : "sb_ajle",
    "torkel" : "sb_rpi4",
    "svea" : "sb_dh1t"
}
STATIONS = {
    "svea" : "api-sveav59-kvartar",
    "torkel" : "api-torkel-kvartar",
    "linkoping" : "api-linkoping-kvartar"
}
SENSORS_REPORT = {
    "linkoping1" : "LKPG_Strandv_1",
    "linkoping2" : "LKPG_Strandv_2",
    "torkel" : "sb_rpi4",
    "svea" : "sb_dh1t"
}

#Downloading sensor data

def get_all_ref(start,end):
    for station in STATIONS.keys():
        api.get_ref_data(station,start,end, FILEPATH_REF + station + "_ref")


def get_all_sensor(start,end,gasses):

    reports = {}
    for sensor,installid in SENSORS_REPORT.items():
        reports[sensor] = api.get_report_data_raw(start,end,installid)
        print("Report data collected for: " + sensor)

    for gas in gasses:
        for sensor in SENSORS.keys():
            for attempt in range(3):
                try:
                    dataset = api.get_one_gas_long(start,end,SENSORS[sensor],gas)
                except:
                    print("Failed to get data for " + sensor + " attempt " + str(attempt+1) + " out of 3")
                    continue
                else:
                    file = FILEPATH_SENS + sensor + "_" + gas
                    dataset = pd.concat([dataset,reports[sensor]],axis=1)
                    dataset = dataset.dropna(thresh=(len(reports[sensor].columns)+1))
                    dataset.to_csv(file + ".csv")
                    print("data saved to: " + file + ".csv")
                    break
                


start = "2025-01-01"
end = "2025-06-30   "
gasses = ["NO","NO2"]

get_all_ref(start,end)
get_all_sensor(start,end,gasses)
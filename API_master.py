import API_tools as api

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

#Downloading sensor data

def get_all_ref(start,end):
    for station in STATIONS.keys():
        api.get_ref_data(station,start,end, FILEPATH_REF + station + "_ref")


def get_all_sensor(start,end,gasses):
    for gas in gasses:
        for sensor in SENSORS.keys():
            for attempt in range(3):
                try:
                    api.get_one_gas_long(start,end,SENSORS[sensor],gas,FILEPATH_SENS + sensor + "_" + gas)
                except:
                    print("Failed to get data for " + sensor + " attempt " + str(attempt+1) + " out of 3")
                    continue
                else:
                    break


start = "2025-01-01"
end = "2025-06-30   "
gasses = ["NO","NO2"]

get_all_ref(start,end)
get_all_sensor(start,end,gasses)
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
            info = {"endpoint_name" : SENSORS[sensor],
            "from" : start,
            "to" : end,
            }
            for attempt in range(3):
                try:
                    api.get_one_gas(info,gas,FILEPATH_SENS + sensor + "_" + gas)
                except:
                    continue
                else:
                    break

start = "2025-03-01"
end = "2025-06-29"
gasses = ["NO","NO2"]

get_all_ref(start,end)
get_all_sensor(start,end,gasses)
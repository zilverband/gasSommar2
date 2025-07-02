"""Contains the class KalmanCalibrator, used to do a baseline calibration of sensor data"""
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#sensor containing only the gas value of interest!
class KalmanCalibrator:
    """ Contains a Kalman filter for tracking the baseline from a sensor.

    The time update step assumes the sensor is constant, with variance Q
    The measurement update step is provided with m_func

    Attributes:
    Q : number, the variance of the baseline per day (model parameter)
    x0 : initial value
    P0 : initial variance
    step : size of time step (type timedelta)
    backwards : False if we calibrate forwards True if backwards
    baseline : expected baseline

    Methods:
    time_update() : performs a time update step
    measure_update(): performs a measurement update step
    run(end=None,verbal=False) : runs the filter to point end 
                        (end of data if end=None), prints updates if verbal=True
    return_calibrated_series : returns calibrated data as a pandas Series

    Example: runs the calibrator and plots data
    measure - some function, takes a pandas series with sensor data and a time as argument
    sensor - pandas series with sensor data

    calibrator = KalmanCalibrator(measure, sensor)
    calibrator.run()
    calibrator.plot()
    plt.show()
    """

    def __init__(self,m_func, sensor ,Q=1, x0=0, P0=1, step=timedelta(days=1),
                  backwards=False, baseline = 0):
        #Defines all Kalman filter variables
        self.Q = Q*step.total_seconds()/86400 #Scaling the time variance with time
        self.m_func = m_func #function that returns y and R
        self.x = x0
        self.P = P0
        self.t = sensor.index[0].round("D")
        self.step = step
        
        self.sensor = sensor #saves the sensor data

        self.backwards = backwards
        self.offset = pd.DataFrame(index=sensor.index, columns=["offset","variance"])
        self.baseline = baseline

    def time_update(self):
        #Kalman formula
        self.x = self.x
        self.P = self.P + self.Q

        #Saves the offset
        if not self.backwards:
            self.offset.loc[self.t:(self.t+self.step),"offset"] = self.x
            self.offset.loc[self.t:(self.t+self.step),"variance"] = self.P
        
        self.t = self.t + self.step

    def measurement_update(self):  
        y, R = self.m_func(self.sensor,self.t)
        if np.isnan(y) or np.isnan(R):
            return None

        #Kalman formula
        x = self.x
        P = self.P
        self.x = x + P/(P+R)*(y-x)
        self.P = P - P**2/(P+R)

        #Saves the offset
        if self.backwards:
            self.offset.loc[(self.t - self.step):self.t,"offset"] = self.x
            self.offset.loc[(self.t - self.step):self.t,"variance"] = self.P
    
    def run(self,end=None, verbal=False):
        #If end isn't specified, run until we're out of data
        if end is None:
            end = self.sensor.index[-1]
        
        #Update step
        while self.t < end:
            self.time_update()
            self.measurement_update()
            if verbal:
                self.print_state()
    
    def return_calibrated_series(self,truncate = True):
        cal_data = self.sensor-self.offset["offset"] + self.baseline
        
        if truncate:
            cal_data[cal_data < 0] = 0
        
        return cal_data

    def plot_offset(self, confidence=True, color = "blue"):
        if confidence:
            std = np.sqrt(self.offset["variance"].astype("Float64"))
            ci = 1.96 * std

            self.offset["offset"].plot(color=color)
            (self.offset["offset"]+ ci).plot(color=color,style = "--")
            (self.offset["offset"]-ci).plot(color=color,style = "--")
        else:
            self.offset["offset"].plot(color=color)

    def plot(self,color="green"):
        self.return_calibrated_series().plot(color=color)

    def plot_raw(self,color="yellow"):
        self.sensor.plot(color=color)

    def print_state(self):
        print("x: {0} P: {1} t: {2}".format(self.x,self.P,self.t))
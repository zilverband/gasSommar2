import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np

#sensor containing only the gas value of interest!
class KalmanCalibrator:
    def __init__(self,Q,m_func,sensor,delta=timedelta(days=7),look_back=None,x0=0,P0=1, backwards=False):
        self.Q = Q* (delta.days ** 3) #Number
        self.m_func = m_func #function that returns y and R
        self.x = x0 #Number
        self.P = P0 #Number
        self.sensor = sensor #pd.Series
        self.delta = delta #timedelta
        self.backwards = backwards #Bool
        self.t = self.sensor.index[0].round("D") #datetime
        self.offset = pd.DataFrame(index=sensor.index, columns=["offset","variance"])
        
        if look_back is None:
            self.look_back = delta
        else:
            self.look_back = look_back

    def time_update(self):
        self.x = self.x
        self.P = self.P + self.Q

        if not self.backwards:
            self.offset.loc[self.t:(self.t+self.delta),"offset"] = self.x
            self.offset.loc[self.t:(self.t+self.delta),"variance"] = self.P
        
        self.t = self.t + self.delta

    def measurement_update(self):
        #Check a week rolling average
        curr_time = self.sensor[(self.t - self.look_back):self.t]
        if curr_time.empty:
            return None
        y, R = self.m_func(curr_time)
        if np.isnan(y) or np.isnan(R):
            return None

        x = self.x
        P = self.P
        self.x = x + P/(P+R)*(y-x)
        self.P = P - P**2/(P+R)
        
        if self.backwards:
            self.offset.loc[(self.t - self.delta):self.t,"offset"] = self.x
            self.offset.loc[(self.t - self.delta):self.t,"variance"] = self.P
    
    def run(self,end=None, verbal=False):
        if end is None:
            end = self.sensor.index[-1]
        
        while self.t < end:
            self.time_update()
            self.measurement_update()
            if verbal:
                self.print_state()
    
    def return_calibrated_series(self,truncate = True):
        cal_data = self.sensor-self.offset["offset"]
        if truncate:
            cal_data[cal_data < 0] = 0
        return cal_data

    def plot_offset(self, confidence=True):
        if confidence:
            std = np.sqrt(self.offset["variance"].astype("Float64"))
            ci = 1.96 * std

            self.offset["offset"].plot(color="blue")
            (self.offset["offset"]+ ci).plot(color="blue",style = "--")
            (self.offset["offset"]-ci).plot(color="blue",style = "--")
        else:
            self.offset["offset"].plot(color="blue")

    def plot(self):
        self.sensor.plot()
        self.return_calibrated_series().plot()
        plt.show()
        self.plot_offset()
        plt.show()

    def print_state(self):
        print("x: {0} P: {1} t: {2}".format(self.x,self.P,self.t))
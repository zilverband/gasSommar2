import load_data as ld
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np

def measurement(data,plot=False):
    N = len(data)
    num_of_base_points = int(N/10)
    outliers = int(num_of_base_points/3)
    
    base_points = data.nsmallest(num_of_base_points)[outliers:]
    y = base_points.mean()
    R= base_points.var()

    if plot:
        data.plot()
        plt.scatter(base_points.index,base_points)
        plt.show()

    return y, R

#sensor containing only the gas value of interest!
class KalmanCalibrator:
    def __init__(self,Q,m_func,sensor,delta=timedelta(days=7),x0=0,P0=1, backwards=False):
        self.Q = Q* (delta.days ** 2) #Number
        self.m_func = m_func #function that returns y and R
        self.x = x0 #Number
        self.P = P0 #Number
        self.sensor = sensor #pd.Series
        self.delta = delta #timedelta
        self.backwards = backwards #Bool
        self.t = self.sensor.index[0].round("D") #datetime
        self.offset = pd.DataFrame(index=sensor.index, columns=["offset","variance"])

    def time_update(self):
        self.x = self.x
        self.P = self.P + self.Q

        if not self.backwards:
            self.offset.loc[self.t:(self.t+self.delta),"offset"] = self.x
            self.offset.loc[self.t:(self.t+self.delta),"variance"] = self.P
        
        self.t = self.t + self.delta

    def measurement_update(self):
        y, R = self.m_func(self.sensor[(self.t - self.delta):self.t])
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
    
    def return_calibrated_series(self):
        return self.sensor-self.offset["offset"]

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
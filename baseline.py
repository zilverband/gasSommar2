import load_data as ld
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
from unit_conversion import df_ugm3_to_ppb

def lowest_points(data,ratio,tresh,plot=True,return_points=False):
    if data.empty:
        return None
    
    N = len(data)
    num_of_base_points = int(N*ratio)
    outliers = int(num_of_base_points*tresh)
    
    base_points = data.nsmallest(num_of_base_points)[outliers:]

    if plot:
        plt.scatter(base_points.index,base_points,color="blue",zorder=5)
        plt.hlines(y=base_points.mean(),xmin=data.index[0],xmax=data.index[-1],color="blue")

    if return_points:
        return base_points
    
    y = base_points.mean()
    R = base_points.var()
    return y,R

def lowest_points_interval(data,delta,ratio,tresh,plot=True,return_points=False):
    if data.empty:
        return None
    
    start = data.index[0]
    end = data.index[-1]
    base_points=None
    
    while start < end:
        new_points = lowest_points(data[start:(start+delta)],ratio,tresh,plot=False,return_points=True)
        if base_points is None:
            base_points= new_points
        else:
            base_points = pd.concat([base_points,new_points])
        start += delta

    base_points = hampel_filter(base_points)
    base_points = split_in_two(base_points)
    
    if plot:
        plt.scatter(base_points.index,base_points,color="blue",zorder=5)
        plt.hlines(y=base_points.mean(),xmin=data.index[0],xmax=end,color="blue")

    if return_points:
        return base_points
    

    y = base_points.mean()
    R= base_points.var()
    return y,R

def hampel_filter(data,n_sigmas=2):
    mu = data.mean()
    sigma = data.std()

    return data[np.abs(data-mu) < n_sigmas*sigma]

def split_in_two(data,tresh=10):
    mean = data.mean()

    data1 = data[data<mean]
    data2 = data[data>mean]
    test = np.abs(data1.mean()-data2.mean())
    if test > tresh:
        if data1.index.mean() > data2.index.mean():
            return data1
        else:
            return data2
    else:
        return data
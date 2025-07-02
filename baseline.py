import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def lowest_points(data, curr, delta, period = None, ratio=1/20, tresh=0,
                   plot=False, return_points=False,filters=["hampel","split"]):
    """Returns a baseline estimate based on the lowest points

    Parameters:
    data : Series - containing sensor data
    curr : datetime - current time
    delta : timedelta - how far back we want to look
    period : timedelta - if not None takes the lowest point in each period
    ratio : ratio of points to be considered low
    tresh : if non-zero removes the lowest points (to omit outliers)
    plot : True,false if we should plot the baseline and points
    return_points : if we should return the points (not y,R)
    """
    data = data[(curr-delta):curr]
    #Exits code if there is no data (otherwise it will crash)
    if data.empty:
        return np.nan, np.nan
    
    if period is None:
        period = delta
    
    #Calculates base point for each interval
    start = data.index[0]
    end = data.index[-1]
    base_points=None

    while start < end:

        curr_data = data[start:(start+period)]
        N = len(curr_data)
        num_of_base_points = int(N*ratio)
        outliers = int(num_of_base_points*tresh)
        new_points = curr_data.nsmallest(num_of_base_points)[outliers:]
        
        if base_points is None:
            base_points= new_points
        else:
            base_points = pd.concat([base_points,new_points])
        start += period

    #Does filtering on the base points
    if "hampel" in filters:
        base_points = hampel_filter(base_points)
    if "split" in filters:
        base_points = split_in_two(base_points)
    if return_points:
        return base_points

    if plot:
        plt.scatter(base_points.index,base_points,color="blue",zorder=5)
        plt.hlines(y=base_points.mean(),xmin=data.index[0],
                   xmax=end,color="blue")

    #Calculates mean and variance of base points
    y = base_points.mean()
    R = base_points.var()
    return y,R

def hampel_filter(data,n_sigmas=2):
    #Simple hampel filter
    mu = data.mean()
    sigma = data.std()
    return data[np.abs(data-mu) < n_sigmas*sigma]

def split_in_two(data,tresh=10):
    mean = data.mean()

    data1 = data[data<mean]
    data2 = data[data>mean]
    test = np.abs(data1.mean()-data2.mean())

    # if we can split it into two parts that are more than tresh ppb 
    # away from each other, use only the last part
    if test > tresh:
        if data1.index.mean() > data2.index.mean():
            return data1
        else:
            return data2
    else:
        return data

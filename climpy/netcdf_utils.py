__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

import netCDF4
import numpy as np


def generate_netcdf_uniform_time_data(time_variable, td1=None, td2=None):
    timeLength = time_variable.shape[0]
    if ( td1 == None):
        td1 = netCDF4.num2date(time_variable[0], time_variable.units)
    if (td2 == None):
        td2 = netCDF4.num2date(time_variable[1], time_variable.units)
    timeStep = td2 - td1
    rawTimeData = np.empty((timeLength,), dtype=object)
    for timeIndex in range(timeLength):
        rawTimeData[timeIndex] = td1 + timeStep * timeIndex

    return rawTimeData
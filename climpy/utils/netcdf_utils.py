import netCDF4
import numpy as np

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'


def generate_netcdf_uniform_time_data(time_variable, td1=None, td2=None):
    time_length = time_variable.shape[0]
    if td1 == None:
        td1 = netCDF4.num2date(time_variable[0], time_variable.units)
    if td2 == None:
        td2 = netCDF4.num2date(time_variable[1], time_variable.units)
    time_step = td2 - td1

    generated_time_data = np.empty((time_length,), dtype=object)
    for timeIndex in range(time_length):
        generated_time_data[timeIndex] = td1 + time_step * timeIndex

    return generated_time_data


def convert_time_data_impl(nc_time_var, time_units):
    time_data = netCDF4.num2date(nc_time_var[:], time_units, only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    return time_data
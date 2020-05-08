from datetime import datetime
import numpy as np
import climpy.netcdf_utils as nc_utils
# from libs.readers import AbstractNetCdfReader as ancr

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

Z_DIM_NC_KEY = 'bottom_top'


def derive_wrf_net_flux_from_accumulated_value(nc, acc_down_name, acc_up_name, tyxSlicesArray, dt):
    down_accumulated = nc.variables[acc_down_name][tyxSlicesArray]
    up_accumulated = nc.variables[acc_up_name][tyxSlicesArray]

    down = down_accumulated[1:] - down_accumulated[:-1]
    up = up_accumulated[1:] - up_accumulated[:-1]

    # convert J to W
    dt_second = dt.days*24*60*60+dt.seconds
    down /= dt_second
    up /= dt_second
    net = down - up

    # net_masked = np.ma.array(np.empty(down_accumulated.shape))
    # net_masked[0] = np.NaN
    # net_masked.mask = land_mask[np.newaxis, :, :]
    # net_masked.data[1:] = net

    return net, down, up


def derive_wrf_net_flux_from_instanteneous_value(nc, down_name, up_name, tyxSlicesArray, dt): #, land_mask, daily_mean_number_of_steps):
    down_flux = nc.variables[down_name][tyxSlicesArray]
    up_flux = nc.variables[up_name][tyxSlicesArray]
    net_flux = down_flux - up_flux

    # net_masked = np.ma.array(net_flux)
    # net_masked.mask = land_mask[np.newaxis, :, :]

    # down_masked = np.ma.array(down_flux)
    # down_masked.mask = land_mask[np.newaxis, :, :]

    # up_masked = np.ma.array(up_flux)
    # up_masked.mask = land_mask[np.newaxis, :, :]

    # return compute_wrf_daily_data(net_masked, daily_mean_number_of_steps), compute_wrf_daily_data(down_masked, daily_mean_number_of_steps), compute_wrf_daily_data(up_masked, daily_mean_number_of_steps)

    return net_flux, down_flux, up_flux


def compute_wrf_daily_data(data, daily_mean_number_of_steps):
    data = data.reshape((-1, daily_mean_number_of_steps) + data.shape[1:])
    data = np.nanmean(data, axis=1)
    return data


def prepare_wrf_net_flux(nc, acc_down_name, acc_up_name, tyxSlicesArray, land_mask, daily_mean_number_of_steps):
    down = nc.variables[acc_down_name][tyxSlicesArray]
    up = nc.variables[acc_up_name][tyxSlicesArray]
    net = down - up
    # convert J to W
    net /= 3 * 60 * 60

    net = compute_wrf_daily_data(net, daily_mean_number_of_steps)
    net = np.ma.array(net)
    net.mask = land_mask[np.newaxis, :, :]

    return net


def generate_netcdf_uniform_time_data(time_variable, td1=None, td2=None):
    if td1 is None:
        # td1 = dt.datetime.strptime(str(netCDF4.chartostring(time_variable[0])), '%Y-%m-%d_%H:%M:%S')
        td1 = datetime.strptime(''.join([char.decode("utf-8") for char in time_variable[0]]), '%Y-%m-%d_%H:%M:%S')
    if td2 is None:
        td2 = datetime.strptime(''.join([char.decode("utf-8") for char in time_variable[1]]), '%Y-%m-%d_%H:%M:%S')
    rawTimeData = nc_utils.generate_netcdf_uniform_time_data(time_variable, td1, td2)

    return rawTimeData


def generate_xarray_uniform_time_data(time_variable, td1=None, td2=None):
    if td1 is None:
        td1 = datetime.strptime(time_variable[0].values.tostring().decode("utf-8"), '%Y-%m-%d_%H:%M:%S')
    if td2 is None:
        td2 = datetime.strptime(time_variable[1].values.tostring().decode("utf-8"), '%Y-%m-%d_%H:%M:%S')
    rawTimeData = nc_utils.generate_netcdf_uniform_time_data(time_variable, td1, td2)

    return rawTimeData
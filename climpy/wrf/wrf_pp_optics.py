import netCDF4
import numpy as np
import os
import pandas as pd
import wrf as wrf
from climpy.utils.atmos_utils import compute_column_from_vmr_profile
import xarray as xr
import argparse
from climpy.utils.wrf_chem_utils import get_aerosols_keys
from climpy.utils.wrf_chem_made_utils import get_aerosols_stack, get_wrf_size_distribution_by_modes
from climpy.utils.refractive_index_utils import mix_refractive_index
import climpy.utils.mie_utils as mie

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

'''
This script derives optical properties for WRF output 
Currently only column AOD for MADE only (log-normal pdfs)
'''

parser = argparse.ArgumentParser()
parser.add_argument("--mode", help="pycharm")
parser.add_argument("--port", help="pycharm")
parser.add_argument("--wrf_in", help="wrf input file path", default='/work/mm0062/b302074/Data/AirQuality/AQABA/chem_100_v22/output/pp_wrf_optics/merge/wrfout_d01_monmean_regrid')
parser.add_argument("--wrf_out", help="wrf output file path", default='/work/mm0062/b302074/Data/AirQuality/AQABA/chem_100_v22/output/pp_wrf_optics/merge/wrfout_d01_monmean_regrid_optics')
args = parser.parse_args()

print('Will process this WRF:\nin {}\nout {}'.format(args.wrf_in, args.wrf_out))

chem_opt = 100

# setup wl grid for optical props
default_wrf_wavelengths = np.array([0.3, 0.4, 0.5, 0.6, 0.999])  # default WRF wavelengths in SW
maritime_wavelengths = np.array([380, 440, 500, 675, 870])/10**3  # aeronet
wavelengths = np.unique(np.append(default_wrf_wavelengths, maritime_wavelengths))  # Comprehensive wl grid

wavelengths = np.array([0.55, ])  # short & quick
#%%


def export_to_netcdf(var, mode='a', is_var_time_dependent=True):
    # have to remove attributes so that xarray can save data to netcdf
    if 'projection' in var.attrs:
        del var.attrs['projection']
    if 'coordinates' in var.attrs:
        del var.attrs['coordinates']

    mode = 'a'  # set mode to append, time needs to be record dimensions (forced by unlimited) to allow ncrcat
    if not os.path.exists(args.wrf_out):
        mode = 'w'

    unlimited_dim = None
    if is_var_time_dependent:
        unlimited_dim = xr_in['so4aj'].dims[0]  # time

    # load should fix the slow writing
    var.load().to_netcdf(args.wrf_out, mode=mode, unlimited_dims=unlimited_dim, format='NETCDF4_CLASSIC')


nc_in = netCDF4.Dataset(args.wrf_in)
xr_in = xr.open_dataset(args.wrf_in)
#%% Derive spectral column AOD
# for now we will process everything in one go, may require a lot of memory
# time_index = None  # equivalent to wrf.ALL_TIMES

dA_vo = get_wrf_size_distribution_by_modes(nc_in, moment='dA', sum_up_modes=True, column=True)
ri_vo = mix_refractive_index(nc_in, chem_opt, wavelengths)  # volume weighted RI


#%% serial implementation


def derive_aerosols_optical_properties(ri_vo, dA_vo):
    '''
    Use this for a single aerosols type and loop through the list
    Currently only extinction / optical depth

    :param ri_vo: RI of the aerosols
    :param dA_vo: cross-section area distribution
    :return:
    '''

    # Compute Mie extinction coefficients
    # dims are time, r, wl
    qext = np.zeros(dA_vo['data'].shape + ri_vo['wl'].shape)
    qext[:] = np.NaN
    for time_index in range(qext.shape[0]):
        for lat_index in range(qext.shape[1]):  # range(5): #
            print(lat_index)
            for lon_index in range(qext.shape[2]):
                # ri, r_data, wavelength = ri_vo['ri'][time_index], dA_vo['radii'], ri_vo['wl']
                if not ri_vo['ri'].mask[time_index, lat_index, lon_index]:
                    mie_vo = mie.get_mie_efficiencies(ri_vo['ri'][time_index, lat_index, lon_index], dA_vo['radii'], ri_vo['wl'])
                    qext[time_index, lat_index, lon_index] = np.swapaxes(mie_vo['qext'], 0, 1)

    # dims: time, r, wl & time, r
    integrand = qext * dA_vo['data'][..., np.newaxis]
    column_od = np.trapz(integrand, np.log(dA_vo['radii']), axis=-2)  # sd is already dAdlnr
    # column_od = np.sum(column_od_by_modes, axis=1)  # sum up modes

    return column_od


column_od = derive_aerosols_optical_properties(ri_vo, dA_vo)

print('DONE')
exit()

# code below is not finished

#%% Export

var_key = 'AOD'
dims = (xr_in['ALT'].dims[0],) + xr_in['ALT'].dims[2:] + ('wavelength', )  # Unlimited dimension has to be first
data = xr.Dataset({var_key: (dims, column_od)})

data[var_key].attrs["long_name"] = 'Spectral column AOD'
data[var_key].attrs["units"] = ""
data[var_key].attrs["description"] = "Derived using Mie and volume weighted RIs"
export_to_netcdf(data)

var_key = 'wavelengths'  # export wavelength grid
dims = ('wavelength', )
data = xr.Dataset({var_key: (dims, wavelengths)})
data[var_key].attrs["long_name"] = 'Wavelengths'
data[var_key].attrs["units"] = "um"
data[var_key].attrs["description"] = "Rho grid (centers of the interval)"
export_to_netcdf(data, is_var_time_dependent=False)

print('PMs are done, proceed to coordinate variables')
# export_to_netcdf(wrf_ds[['XTIME', 'XLONG', 'XLAT']])  # export time/lat/lon as well
export_to_netcdf(xr_in[['XTIME', 'lon', 'lat']])  # export time/lat/lon as well

print('DONE')


#%% ray implementation

#
# def thread_safe_calculations(ri, r_data, wavelength):  # function for ray
#     mie_vo = mie.get_mie_efficiencies(ri, r_data, wavelength)
#     return np.swapaxes(mie_vo['qext'], 0, 1)  # qext[time_index, lat_index, lon_index]
#
#
# def derive_aerosols_optical_properties(ri_vo, dA_vo):
#     '''
#     Use this for a single aerosols type and loop through the list
#     Currently only extinction / optical depth
#
#     :param ri_vo: RI of the aerosols
#     :param dA_vo: cross-section area distribution
#     :return:
#     '''
#
#     # Compute Mie extinction coefficients
#     # dims are time, r, wl
#     qext = np.zeros(dA_vo['data'].shape + ri_vo['wl'].shape)
#
#     for time_index in range(qext.shape[0]):
#         print('time index: {}'.format(time_index))
#         for lat_index in qext.shape[1]:  # range(5): #
#             print(lat_index)
#             output_ids = []
#             for lon_index in range(qext.shape[2]):
#                 ri, r_data, wavelength = ri_vo['ri'][time_index, lat_index, lon_index], dA_vo['radii'], ri_vo['wl']
#                 # mie_vo = mie.get_mie_efficiencies(ri_vo['ri'][time_index, lat_index, lon_index], dA_vo['radii'], ri_vo['wl'])
#                 # qext[time_index, lat_index, lon_index] = np.swapaxes(mie_vo['qext'], 0, 1)
#                 output_ids.append(thread_safe_calculations.remote(ri, r_data, wavelength))
#
#             output_list = ray.get(output_ids)
#             qext[time_index, lat_index, :] = np.array(output_list)
#
#     # dims: time, r, wl & time, r
#     integrand = qext * dA_vo['data'][..., np.newaxis]
#     column_od = np.trapz(integrand, np.log(dA_vo['radii']), axis=-2)  # sd is already dAdlnr
#     # column_od = np.sum(column_od_by_modes, axis=1)  # sum up modes
#
#     return column_od
#
#
# column_od = derive_aerosols_optical_properties(ri_vo, dA_vo)

#%% Test ray
#
# import ray
# ray.init()
#
# @ray.remote
# def f(x):
#     return x * x
#
# futures = [f.remote(i) for i in range(4)]
# print(ray.get(futures)) # [0, 1, 4, 9]
#
# @ray.remote
# class Counter(object):
#     def __init__(self):
#         self.n = 0
#
#     def increment(self):
#         self.n += 1
#
#     def read(self):
#         return self.n
#
# counters = [Counter.remote() for i in range(4)]
# [c.increment.remote() for c in counters]
# futures = [c.read.remote() for c in counters]
# print(ray.get(futures)) # [1, 1, 1, 1]
import netCDF4
import os
import numpy as np
import xarray as xr
import wrf as wrf
from climpy.utils.atmos_utils import compute_column_from_vmr_profile
import argparse

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

'''
Script derives several common diagnostics from WRF output, such as SO2 & O3 columns in DU
'''

parser = argparse.ArgumentParser()
parser.add_argument("--mode", help="pycharm")
parser.add_argument("--port", help="pycharm")
parser.add_argument("--wrf_in", help="wrf input file path")   # , default='/work/mm0062/b302074/Data/AirQuality/AQABA/chem_100_v22/output/wrfout_d01_2017-06-15_00:00:00')
parser.add_argument("--wrf_out", help="wrf output file path")  # , default='/work/mm0062/b302074/Data/AirQuality/AQABA/chem_100_v22/output/pp_wrf/wrfout_d01_2017-06-15_00:00:00')
args = parser.parse_args()

wrf_in_file_path = args.wrf_in
wrf_out_file_path = args.wrf_out

print('Will process this WRF:\nin {}\nout {}'.format(wrf_in_file_path, wrf_out_file_path))

#%%


def export_to_netcdf(var, mode='a'):
    # have to remove attributes so that xarray can save data to netcdf
    if 'projection' in var.attrs:
        del var.attrs['projection']
    if 'coordinates' in var.attrs:
        del var.attrs['coordinates']
    # set mode to append, time needs to be record dimensions (forced by unlimited) to allow ncrcat
    mode = 'a'  # set mode to append, time needs to be record dimensions (forced by unlimited) to allow ncrcat
    if not os.path.exists(wrf_out_file_path):
        mode = 'w'

    unlimited_dim = xr_in['so4aj'].dims[0]  # Time
    # load should fix the slow writing
    var.load().to_netcdf(wrf_out_file_path, mode=mode, unlimited_dims=unlimited_dim, format='NETCDF4_CLASSIC')


nc_in = netCDF4.Dataset(wrf_in_file_path)
xr_in = xr.open_dataset(wrf_in_file_path, engine='netcdf4')

# index of the vertical dimensions
z_dim_axis = 1

# for now we will process everything in one go, may require a lot of memory
# time_index = None  # equivalent to wrf.ALL_TIMES

# step by step processing
# n_time_steps = nc_in.dimensions['Time'].size
# for time_index in range(0, n_time_steps):
time_index = wrf.ALL_TIMES  # process all in one go
# print('processing step {} out of {}'.format(time_index, n_time_steps))
# wrf_out_file_path = wrf_in_file_path + '_pp_{}'.format(time_index)  # comes from the arguments
print('output will be saved into {}'.format(wrf_out_file_path))

slp = wrf.getvar(nc_in, "slp", timeidx=time_index, squeeze=False)
export_to_netcdf(slp)  # , mode='w')

pressure = wrf.getvar(nc_in, "pressure", timeidx=time_index, squeeze=False)
export_to_netcdf(pressure)

z_stag = wrf.getvar(nc_in, "zstag", timeidx=time_index, squeeze=False)
z_stag.name = 'height_stag'
export_to_netcdf(z_stag)

z_rho = wrf.getvar(nc_in, "z", timeidx=time_index, squeeze=False)
export_to_netcdf(z_rho)

# compute dP
p_sfc = wrf.getvar(nc_in, 'PSFC', timeidx=time_index, squeeze=False) / 10**2
pressure_stag = np.zeros(z_stag.shape)
pressure_stag[:, 0] = p_sfc
for layer_index in range(0, pressure.shape[z_dim_axis]):
    half_dp = pressure_stag[:, layer_index] - pressure[:, layer_index]
    pressure_stag[:, layer_index+1] = pressure_stag[:, layer_index] - half_dp * 2

dp = pressure.copy()
dp.name = 'dP'
dp.attrs['description'] = 'layers thickness dP'
dp[:] = -1 * np.diff(pressure_stag, axis=z_dim_axis)
export_to_netcdf(dp)

# dZ
dz = dp.copy()
dz.name = 'dZ'
dz.attrs['description'] = 'layers thickness dZ'
dz.attrs['units'] = 'm'
dz[:] = np.diff(z_stag, axis=z_dim_axis)
export_to_netcdf(dz)

# compute center of mass for so2 and sulf
for var_key in ('so2', 'sulf'):
    gas_ppmv = wrf.getvar(nc_in, var_key, timeidx=time_index, squeeze=False)
    gas_cm = slp.copy()
    gas_cm.name = '{}_mc'.format(var_key)
    gas_cm.attrs['units'] = 'm'
    gas_cm.attrs['description'] = '{} center of mass'.format(var_key)
    gas_cm[:] = np.sum(gas_ppmv * z_rho * dp, axis=z_dim_axis) / np.sum(gas_ppmv * dp, axis=z_dim_axis)
    export_to_netcdf(gas_cm)

    # compute global data
    gas_cm_global = np.sum(gas_ppmv * z_rho * dp, axis=(1, 2, 3)) / np.sum(gas_ppmv * dp, axis=(1, 2, 3))
    gas_cm_global.name = '{}_mc_global'.format(var_key)
    gas_cm_global.attrs['untis'] = 'm'
    gas_cm_global.attrs['decription'] = 'global center of mass for {}'.format(var_key)
    export_to_netcdf(gas_cm_global)

# compute the column dobson units for so2 and ozone
temperature = wrf.getvar(nc_in, 'temp', timeidx=time_index, squeeze=False)
for var_key in ('so2', 'o3'):
    gas_ppmv = wrf.getvar(nc_in, var_key, timeidx=time_index, squeeze=False)
    gas_dobson_units = slp.copy()
    gas_dobson_units.name = '{}_DU'.format(var_key)
    gas_dobson_units.attrs['units'] = 'Dobson units'
    gas_dobson_units.attrs['description'] = '{} column in DU'.format(var_key)
    gas_dobson_units[:] = compute_column_from_vmr_profile(pressure * 10 ** 2, temperature, dz, gas_ppmv, z_dim_axis=z_dim_axis)
    export_to_netcdf(gas_dobson_units)

# compute column loading (not Dobson units)
for var_key in ('nh3', ):
    gas_ppmv = wrf.getvar(nc_in, var_key, timeidx=time_index, squeeze=False)
    gas_column = slp.copy()
    gas_column.name = '{}_column'.format(var_key)
    gas_column.attrs['units'] = 'molecules m^{-2}'
    gas_column.attrs['description'] = '{} column for comparison with satellites'.format(var_key)
    gas_column[:] = compute_column_from_vmr_profile(pressure * 10 ** 2, temperature, dz, gas_ppmv, z_dim_axis=z_dim_axis, in_DU=False)
    export_to_netcdf(gas_column)


print('DONE')
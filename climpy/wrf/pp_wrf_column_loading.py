import netCDF4
import os
import numpy as np
import xarray as xr
import wrf as wrf
from climpy.utils.atmos_utils import compute_column_from_vmr_profile
import argparse
from climpy.utils.tropomi_utils import TROPOMI_in_WRF_KEYS

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'


'''
Script derives several common diagnostics from WRF output, such as SO2 & O3 columns in DU

# THOFA run example
# sbatch $BASH_SCRIPTS/pp_wrf_column_loading_lazy_ensemble.sh /scratch/osipovs/Data/AirQuality/THOFA/chem_100_v2025.0

'''

parser = argparse.ArgumentParser()
parser.add_argument("--mode", "--port", "--host", help="pycharm")
parser.add_argument("--wrf_in", help="wrf input file path")#, default='/work/mm0062/b302074/Data/AirQuality/EMME/2017/chem_100_v1/output/wrfout_d01_2017-12-14_00_00_00')
parser.add_argument("--wrf_out", help="wrf output file path")# , default='/work/mm0062/b302074/Data/AirQuality/EMME/2017/chem_100_v1/output/pp_wrf/wrfout_d01_2017-12-14_00_00_00')
args = parser.parse_args()

#%%
# args.wrf_in = '/scratch/osipovs/Data/AirQuality/THOFA/chem_100_v1/wrfout_d01_2023-06-10_00_00_00'
# args.wrf_out = '/scratch/osipovs/Data/AirQuality/THOFA/chem_100_v1/pp/column_loading/wrfout_d01_2023-06-10_00_00_00'
print('Will process this WRF:\nin {}\nout {}'.format(args.wrf_in, args.wrf_out))
#%%


def export_to_netcdf(var, mode='a'):
    # have to remove attributes so that xarray can save data to netcdf
    if 'projection' in var.attrs:
        del var.attrs['projection']
    if 'coordinates' in var.attrs:
        del var.attrs['coordinates']
    # set mode to append, time needs to be record dimensions (forced by unlimited) to allow ncrcat
    mode = 'a'  # set mode to append, time needs to be record dimensions (forced by unlimited) to allow ncrcat
    if not os.path.exists(args.wrf_out):
        mode = 'w'

    unlimited_dim = xr_in['so4aj'].dims[0]  # Time
    # load should fix the slow writing
    var.load().to_netcdf(args.wrf_out, mode=mode, unlimited_dims=unlimited_dim, format='NETCDF4_CLASSIC')


nc_in = netCDF4.Dataset(args.wrf_in)
xr_in = xr.open_dataset(args.wrf_in, engine='netcdf4')

# index of the vertical dimensions
z_dim_axis = 1

# for now we will process everything in one go, may require a lot of memory
# time_index = None  # equivalent to wrf.ALL_TIMES

# step by step processing
# n_time_steps = nc_in.dimensions['Time'].size
# for time_index in range(0, n_time_steps):
time_index = wrf.ALL_TIMES  # process all in one go
# print('processing step {} out of {}'.format(time_index, n_time_steps))
# args.wrf_out = args.wrf_in + '_pp_{}'.format(time_index)  # comes from the arguments
print('output will be saved into {}'.format(args.wrf_out))

pressure = wrf.getvar(nc_in, "pressure", timeidx=time_index, squeeze=False)
temperature = wrf.getvar(nc_in, 'temp', timeidx=time_index, squeeze=False)
z_stag = wrf.getvar(nc_in, "zstag", timeidx=time_index, squeeze=False)
# z_stag.name = 'height_stag'
z_rho = wrf.getvar(nc_in, "z", timeidx=time_index, squeeze=False)
export_to_netcdf(z_rho)

# dZ
dz = z_stag.diff(dim='bottom_top_stag').rename({'bottom_top_stag': 'bottom_top'})  # m
# dz = z_stag.copy()
dz.name = 'dZ'
dz.attrs['description'] = 'layers thickness dZ'
dz.attrs['units'] = 'm'
# dz[:] = np.diff(z_stag, axis=z_dim_axis)

# compute column loading (not Dobson units)
for var_key in TROPOMI_in_WRF_KEYS:
    gas_ppmv = wrf.getvar(nc_in, var_key, timeidx=time_index, squeeze=False)
    gas_column_da = compute_column_from_vmr_profile(pressure * 10 ** 2, temperature, dz, gas_ppmv, z_dim_axis=z_dim_axis, in_DU=False)
    # gas_column = slp.copy()
    gas_column_da.name = '{}_column'.format(var_key)
    gas_column_da.attrs['units'] = 'molecules m^{-2}'
    gas_column_da.attrs['description'] = '{} column for comparison with satellites'.format(var_key)
    export_to_netcdf(gas_column_da)


print('DONE')
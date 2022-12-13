import netCDF4
import numpy as np
import os
import pandas as pd
import wrf as wrf
from climpy.utils.atmos_utils import compute_column_from_vmr_profile
import xarray as xr
import argparse
from climpy.utils.wrf_chem_utils import get_aerosols_keys
from climpy.utils.wrf_chem_made_utils import get_aerosols_stack

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

parser = argparse.ArgumentParser()
parser.add_argument("--mode", help="pycharm")
parser.add_argument("--port", help="pycharm")
parser.add_argument("--wrf_in", help="wrf input file path", default='/work/mm0062/b302074/Data/AirQuality/EMME/chem_100_v100/output/pp_wrf_health/boa/wrfout_d01_2017-06-15_00:00:00')
parser.add_argument("--wrf_out", help="wrf output file path")  # default='/work/mm0062/b302074/Data/AirQuality/AQABA/chem_100_v15/output/pp_wrf_health/pm/wrfout_d01_2017-08-31_00:00:00'
args = parser.parse_args()

print('Will process this WRF:\nin {}\nout {}'.format(args.wrf_in, args.wrf_out))

chem_opt = 100
aerosols_keys = get_aerosols_keys(chem_opt)
#%%


def export_to_netcdf(var, mode='a'):
    # have to remove attributes so that xarray can save data to netcdf
    if 'projection' in var.attrs:
        del var.attrs['projection']
    if 'coordinates' in var.attrs:
        del var.attrs['coordinates']

    mode = 'a'  # set mode to append, time needs to be record dimensions (forced by unlimited) to allow ncrcat
    if not os.path.exists(args.wrf_out):
        mode = 'w'

    unlimited_dim = xr_in['so4aj'].dims[0]  # time
    # load should fix the slow writing
    var.load().to_netcdf(args.wrf_out, mode=mode, unlimited_dims=unlimited_dim, format='NETCDF4_CLASSIC')


nc_in = netCDF4.Dataset(args.wrf_in)
xr_in = xr.open_dataset(args.wrf_in)

# squeeze singleton (level, boa) dimension, due to cdo issues, it can not process 5D output
squeeze_level_dim = True
if squeeze_level_dim:
    xr_in = xr_in.squeeze(dim='bottom_top')  # dims order is inversed in xarray and numpy

# Process everything in one go, may require a lot of memory
# time_index = None  # equivalent to wrf.ALL_TIMES

# compute PMs
pm1_sizes = [1 * 10 ** -20, 1 * 10 ** -6]  # m, min max for integration
pm25_sizes = [1 * 10 ** -20, 2.5 * 10 ** -6]  # m, min max for integration
pm10_sizes = [1 * 10 ** -20, 10 * 10 ** -6]  # m, min max for integration

wrf_pms = {}
pm_keys = ['PM1', 'PM25', 'PM10']
pm_sizes = pm1_sizes
key = pm_keys[0]
for pm_sizes, key in zip([pm1_sizes, pm25_sizes, pm10_sizes], pm_keys):
    print('Processing {}[{} - {}]'.format(key, pm_sizes[0], pm_sizes[1]))
    wrf_diags_vstack, wrf_keys = get_aerosols_stack(nc_in, aerosols_keys, pm_sizes=pm_sizes, combine_modes=True, combine_organics=True, combine_sea_salt=True)
    if squeeze_level_dim:
        wrf_diags_vstack = wrf_diags_vstack.squeeze(axis=2)
    # wrf_diags_vstack = wrf_diags_vstack[..., boa_ind]
    pm = np.sum(wrf_diags_vstack, axis=0)  # this will sum up PM from individual species

    var_key = '{}_by_type'.format(key)  # export by type
    print('diag {} computed, proceed to saving'.format(var_key))
    # dims = ('type',) + wrf_ds['so4aj'].dims
    # data = xr.Dataset({var_key: (dims, wrf_diags_vstack.filled())})
    # Unlimited dimension has to be first
    dims = (xr_in['so4aj'].dims[0],) + ('type',) + xr_in['so4aj'].dims[1:]
    data = xr.Dataset({var_key: (dims, wrf_diags_vstack.filled().swapaxes(0, 1))})

    data[var_key].attrs["long_name"] = '{} by aerosol type'.format(key)
    data[var_key].attrs["units"] = "ug m^-3"
    data[var_key].attrs["description"] = "{} derived by integrating MADE size distribution across all modes: {}".format(key, ','.join(wrf_keys))
    export_to_netcdf(data)

    # export total
    var_key = '{}'.format(key)
    print('diag {} computed, proceed to saving'.format(var_key))
    dims = xr_in['so4aj'].dims
    data = xr.Dataset({var_key: (dims, pm)})
    data[var_key].attrs["long_name"] = '{}'.format(key)
    data[var_key].attrs["units"] = "ug m^-3"
    data[var_key].attrs["description"] = "{} derived by integrating MADE size distribution across all modes".format(key)
    export_to_netcdf(data)

print('PMs are done, proceed to coordinate variables')
# export time/lat/lon as well

export_to_netcdf(xr_in[['XTIME', 'XLONG', 'XLAT']])

#%% Compute the UVI
print('computing UV index')
xr_in.load()  # preload to avoid saving issues (otherwise you get zeros in the netcdf)
xr_in['UVI'] = xr_in['PH_ERYTHEMA']/25 * 10**3  # UVI = 1(25 mW m^-2) * integral (I*w*dlambda)
xr_in.to_netcdf(args.wrf_in)

print('DONE')

import netCDF4
from distutils.util import strtobool
import numpy as np
import os
import pandas as pd
import wrf as wrf
from climpy.utils.atmos_utils import compute_column_from_vmr_profile
import xarray as xr
import argparse
from climpy.utils.wrf_chem_utils import get_aerosols_keys
from climpy.utils.wrf_chem_made_utils import get_aerosols_pm_stack

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

'''
https://pubs.geoscienceworld.org/sepm/jsedres/article/71/3/365/114077/aerodynamic-and-geometric-diameters-of-airborne
Conventionally, an aerodynamic diameter is taken as a product of the geometric diameter and the square root of particle density.


Run like this:
gogomamba
campaign=THOFA
sim_version=chem_100_v8
campaign=AREAD
sim_version=chem_100_v2

local:
wrf_output_folder=/home/osipovs/Data/AirQuality/$campaign/$sim_version/output/
levante:
wrf_output_folder=/work/mm0062/b302074/Data/AirQuality/$campaign/$sim_version/output/

python -u ${CLIMPY}/climpy/wrf/wrf_pp_health.py --wrf_in=$wrf_output_folder/ship_track/wrf_ship_track.nc --wrf_out=$wrf_output_folder/ship_track/pp/wrf_ship_track.nc

'''

parser = argparse.ArgumentParser()
parser.add_argument("--mode", help="pycharm")
parser.add_argument("--port", help="pycharm")
parser.add_argument("--host", help="pycharm")
parser.add_argument("--wrf_in", help="wrf input file path", default='/work/mm0062/b302074/Data/AirQuality/EMME/2050/CLE/chem_100_v1/output/wrfout_d01_2050-12-02_00_00_00')  # default='/work/mm0062/b302074/Data/AirQuality/EMME/chem_100_v100/output/pp_wrf_health/boa/wrfout_d01_2017-06-15_00:00:00')
parser.add_argument("--wrf_out", help="wrf output file path", default='/work/mm0062/b302074/Data/AirQuality/AREAD/chem_100_v1/output/pp_health/wrfout_d01_2022-11-01_00_00_00')
parser.add_argument("--pm_input_is_aerodynamic_diameter", help="True/False", type=strtobool, default=True)  # default='/work/mm0062/b302074/Data/AirQuality/AQABA/chem_100_v15/output/pp_wrf_health/pm/wrfout_d01_2017-08-31_00:00:00'
args = parser.parse_args()

print('Will process this WRF:\nin {}\nout {}'.format(args.wrf_in, args.wrf_out))
print('pm_input_is_aerodynamic_diameter is:{}'.format(args.pm_input_is_aerodynamic_diameter))
if args.pm_input_is_aerodynamic_diameter:
    print('aerodynamic')

chem_opt = 100
aerosols_keys = get_aerosols_keys(chem_opt)
#%%
xr_in = xr.open_dataset(args.wrf_in)
xr_in_boa = xr_in.sel(bottom_top=0) #.sel(Time=[0,1,2])  # DEBUG only

# compute PMs
pm1_sizes = [1 * 10 ** -20, 1 * 10 ** -6]  # m, min max for integration
pm25_sizes = [1 * 10 ** -20, 2.5 * 10 ** -6]  # m, min max for integration
pm10_sizes = [1 * 10 ** -20, 10 * 10 ** -6]  # m, min max for integration

pms = []
pm_keys = ['PM1', 'PM25', 'PM10']
pm_size_range = pm1_sizes
key = pm_keys[0]
for pm_size_range, key in zip([pm1_sizes, pm25_sizes, pm10_sizes], pm_keys):
    print('Processing {}[{} - {}]'.format(key, pm_size_range[0], pm_size_range[1]))
    pm_ds, wrf_keys = get_aerosols_pm_stack(xr_in_boa, aerosols_keys, pm_input_is_aerodynamic_diameter=args.pm_input_is_aerodynamic_diameter, pm_size_range=pm_size_range, combine_modes=True, combine_organics=True, combine_sea_salt=True)
    pm_by_type_key = '{}_by_type'.format(key)
    if pm_ds.dims[1] == 'Time':  # Reorder dimensions making Time first (to be CDO complaint)
        reordered_dims = (pm_ds.dims[1], pm_ds.dims[0]) + pm_ds.dims[2:]
        pm_ds = pm_ds.transpose(*reordered_dims)
    pm_ds = pm_ds.rename(pm_by_type_key).to_dataset()

    pm_ds[key] = pm_ds[pm_by_type_key].sum(dim='aerosol')  # sum up PM from individual species
    aerosols_string = ', '.join(pm_ds.aerosol.to_numpy().tolist())

    pm_ds[key].attrs["units"] = "ug m^-3"
    pm_ds[key].attrs["description"] = "{} derived by integrating MADE size distribution across all modes and aerosol types".format(key)
    pm_ds[pm_by_type_key].attrs["units"] = "ug m^-3"
    pm_ds[pm_by_type_key].attrs["aerosol"] = aerosols_string  # work around to store the aerosol dimension in CDO complaint way
    pm_ds[pm_by_type_key].attrs["description"] = "{} derived by integrating MADE size distribution across all modes".format(key)

    pms += [pm_ds, ]

wrf_ds = xr.merge(pms)  # merge PM1, PM2.5 and PM10
if args.pm_input_is_aerodynamic_diameter:
    wrf_ds.attrs["description"] = "PM size is defined as aerodynamic diameter"
else:
    wrf_ds.attrs["description"] = "PM size is defined as geometric diameter"

#%% Compute the UVI
print('computing UV index')
if 'PH_ERYTHEMA' in xr_in_boa.variables.keys():
    wrf_ds['UVI'] = xr_in_boa['PH_ERYTHEMA'] / 25 * 10 ** 3  # UVI = 1(25 mW m^-2) * integral (I*w*dlambda)

#%% Make ds CDO complaint
# for var_key in wrf_ds.variables:
#     wrf_ds[var_key].encoding['_FillValue'] = None  # make CDO compliant
#%% export section
if not os.path.exists(os.path.dirname(args.wrf_out)):
    os.makedirs(os.path.dirname(args.wrf_out))

# wrf_ds.to_netcdf(args.wrf_out)  # having an aerosol variable as strings breaks down cdo
wrf_ds.attrs['aerosol'] = aerosols_string  # store the aerosols as attribute as a workaround
wrf_ds['aerosol'] = np.arange(len(wrf_ds.aerosol))
wrf_ds.to_netcdf(args.wrf_out)
#%%
print('DONE')

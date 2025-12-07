import numpy as np
import xarray as xr
import argparse
import pandas as pd
import os
# this will fix the esmf import bug
import os
from pathlib import Path
if 'ESMFMKFILE' not in os.environ:  # os.environ.get('READTHEDOCS') and
    # RTD doesn't activate the env, and esmpy depends on a env var set there
    # We assume the `os` package is in {ENV}/lib/pythonX.X/os.py
    # See conda-forge/esmf-feedstock#91 and readthedocs/readthedocs.org#4067
    print('fixing ESMFMKFILE env variable')
    os.environ['ESMFMKFILE'] = str(Path(os.__file__).parent.parent / 'esmf.mk')
import xesmf as xe

'''
Conservatively regrid Sentinel-5P dataset onto WRF grid 

Run Template:
data_root=/Users/osipovs/Data/Copernicus/Sentinel-5P/
python -u ${CLIMPY}/examples/regridding/regrid_tropomi_on_wrf_grid.py --tropomi_in=$data_root/S5P_OFFL_L2__CH4____20230601T081351_20230601T095521_29183_03_020500_20230603T044522 --tropomi_out=$data_root/d02/S5P_OFFL_L2__CH4____20230601T081351_20230601T095521_29183_03_020500_20230603T044522 --tropomi_key=methane_mixing_ratio_bias_corrected
'''

parser = argparse.ArgumentParser()
parser.add_argument("--mode", help="pycharm")
parser.add_argument("--host", help="pycharm")
parser.add_argument("--port", help="pycharm")
parser.add_argument("--wrf_in", help="wrf geo_em input file path")#, default='/work/mm0062/b302074/Data/AirQuality/AQABA/IC_BC/geo_em.d01.nc')
parser.add_argument("--tropomi_in", help="TROPOMI file path")#, default='/work/mm0062/b302074/Data/NASA/SEDAC/population_density/gpw_v4_population_density_rev11_2pt5_min.nc')  # '/work/mm0062/b302074/Data/NASA/SEDAC/gpw_v4_population_count_adjusted_rev11_2pt5_min.nc')
parser.add_argument("--tropomi_out", help="regridded TROPOMI output file path")#, default='/work/mm0062/b302074/Data/AirQuality/AQABA/IC_BC/aux/gpw_v4_population_density_rev11_2pt5_min.nc_regrid.nc')
parser.add_argument("--tropomi_key", help="TROPOMI variable to regrid")#, default='/work/mm0062/b302074/Data/AirQuality/AQABA/IC_BC/aux/gpw_v4_population_density_rev11_2pt5_min.nc_regrid.nc')
args = parser.parse_args()

#%% local use case for WRF EMME sim & GHS population dataset
# args.wrf_in = '/Users/osipovs/Data/AirQuality/THOFA/IC_BC/geo_em.d02.nc'
# data_root = '/Users/osipovs/Data/Copernicus/Sentinel-5P/'
# args.tropomi_in = data_root + 'S5P_OFFL_L2__CH4____20230601T081351_20230601T095521_29183_03_020500_20230603T044522.nc'  # regional
# args.tropomi_out = data_root + 'd02/S5P_OFFL_L2__CH4____20230601T081351_20230601T095521_29183_03_020500_20230603T044522.nc'
# args.tropomi_key = 'methane_mixing_ratio_bias_corrected'
#
# args.tropomi_in = data_root + 'S5P_OFFL_L2__SO2____20230601T081351_20230601T095521_29183_03_020401_20230603T102223.nc'  # regional
# args.tropomi_out = data_root + 'd02/S5P_OFFL_L2__SO2____20230601T081351_20230601T095521_29183_03_020401_20230603T102223.nc'
# args.tropomi_key = 'sulfurdioxide_total_vertical_column'
#
# args.tropomi_in = data_root + 'S5P_OFFL_L2__HCHO___20230601T081351_20230601T095521_29183_03_020401_20230603T044522.nc'  # regional
# args.tropomi_out = data_root + 'd02/S5P_OFFL_L2__HCHO___20230601T081351_20230601T095521_29183_03_020401_20230603T044522.nc'
# args.tropomi_key = 'formaldehyde_tropospheric_vertical_column'
#
# d01 case
# args.wrf_in = '/Users/osipovs/Data/AirQuality/THOFA/IC_BC/geo_em.d01.nc'
# data_root = '/Users/osipovs/Data/Copernicus/Sentinel-5P/'
# args.tropomi_in = data_root + 'S5P_OFFL_L2__NO2____20230601T081351_20230601T095521_29183_03_020500_20230603T044537.nc'  # regional
# args.tropomi_out = data_root + 'd01/S5P_OFFL_L2__NO2____20230601T081351_20230601T095521_29183_03_020500_20230603T044537.nc'
# args.tropomi_key = 'nitrogendioxide_tropospheric_column'

print('Will regrid this TROPOMI onto this WRF:\nin grid {}\nout grid {}'.format(args.tropomi_in, args.wrf_in))
#%% Build source grid & var. Conservative regridding requires corners
wrf_grid_ds = xr.open_dataset(args.wrf_in)
wrf_grid_ds = wrf_grid_ds.isel(Time=0)
print('Lon[0,0]: {}, Lon[-1,-1]: {}'.format(wrf_grid_ds['XLONG_M'][0,0].item(), wrf_grid_ds['XLONG_M'][-1,-1].item()))  # print rho grid coordinates
print('Lat[0,0]: {}, Lat[-1,-1]: {}'.format(wrf_grid_ds['XLAT_M'][0,0].item(), wrf_grid_ds['XLAT_M'][-1,-1].item()))
xlong_m = wrf_grid_ds['XLONG_M']
xlat_m = wrf_grid_ds['XLAT_M']
xlong_c = wrf_grid_ds['XLONG_C']  # corners are required for conservative remapping
xlat_c = wrf_grid_ds['XLAT_C']

####### Build TROPOMI ds from two NetCDF groups
product_ds = xr.open_dataset(args.tropomi_in, group='PRODUCT')  # TROPOMI

# conservative regridding requires corners.
geo_ds = xr.open_dataset(args.tropomi_in, group='/PRODUCT/SUPPORT_DATA/GEOLOCATIONS')
geo_ds = geo_ds[['latitude_bounds', 'longitude_bounds']].squeeze()
geo_ds = geo_ds.rename({'latitude_bounds': 'lat_b', 'longitude_bounds': 'lon_b'})

# Build the TROPOMI staggered grid
# See figure 6 on page 34 about TROPOMI corners, Start lower left, counterclockwise: https://sentinels.copernicus.eu/documents/247904/2474726/Sentinel-5P-Level-2-Product-User-Manual-Nitrogen-Dioxide.pdf
lat_b = xr.concat((geo_ds.lat_b[:,:, 0], geo_ds.lat_b[:,-1, 1]), dim='ground_pixel')
lon_b = xr.concat((geo_ds.lon_b[:,:, 0], geo_ds.lon_b[:,-1, 1]), dim='ground_pixel')
# append the top pixels
row = xr.concat((geo_ds.lat_b[-1,:, 3], geo_ds.lat_b[-1,-1, 2]), dim='ground_pixel')
lat_b = xr.concat((lat_b, row), dim='scanline')
row = xr.concat((geo_ds.lon_b[-1,:, 3], geo_ds.lon_b[-1,-1, 2]), dim='ground_pixel')
lon_b = xr.concat((lon_b, row), dim='scanline')
lat_b = lat_b.rename({'ground_pixel': 'ground_pixel_stag', 'scanline':'scanline_stag'})
lon_b = lon_b.rename({'ground_pixel': 'ground_pixel_stag', 'scanline':'scanline_stag'})

# Extra Aux Data for post-processing WRF
details_ds = xr.open_dataset(args.tropomi_in, group='/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS')
input_data_ds = xr.open_dataset(args.tropomi_in, group='/PRODUCT/SUPPORT_DATA/INPUT_DATA')

# determine the TROPOMI product and choose the right keys to merge
tropomi_diag = os.path.basename(args.tropomi_in)[11:20].strip('_')  # 'CH4', etc
if tropomi_diag =='CH4':
    print('Processing CH4 switch')
    keys = ['methane_mixing_ratio_bias_corrected', 'qa_value', 'time_utc']
    ds_in = xr.merge([product_ds[keys], lat_b, lon_b, details_ds.column_averaging_kernel, input_data_ds[['methane_profile_apriori', 'altitude_levels', 'pressure_interval', 'surface_pressure']]])
elif tropomi_diag == 'NO2':
    print('Processing NO2 switch')
    keys = ['nitrogendioxide_tropospheric_column', 'qa_value', 'time_utc', 'averaging_kernel', 'air_mass_factor_troposphere', 'air_mass_factor_total', 'tm5_constant_a', 'tm5_constant_b', 'tm5_tropopause_layer_index']
    ds_in = xr.merge([product_ds[keys], lat_b, lon_b, input_data_ds[['surface_pressure']]])
    ds_in = ds_in.set_coords(['tm5_constant_a', 'tm5_constant_b'])  # set as coordinates to avoid loosing them, as they don't depend on lat/lon

ds_in = ds_in.squeeze()  # do the squeeze last after merging
# ds_in = ds_in.assign_coords(time=[pd.to_datetime(ds_in.time_utc.values).mean()])  # original TROPOMI time is only accurate within a day. Take a mean of the exact time across the scanline and add to the dataset
ds_in['time'][...] = pd.to_datetime(ds_in.time_utc.values).mean()

####### DONE TropOMI preparations

ds_out = xr.Dataset({'lat': (['south_north', 'west_east'], xlat_m.data), 'lon': (['south_north', 'west_east'], xlong_m.data),
                     'lat_b': (['south_north_stag', 'west_east_stag'], xlat_c.data), 'lon_b': (['south_north_stag', 'west_east_stag'], xlong_c.data),})  # it is important to call variables lat_b to indicate corners
#%%
regridder = xe.Regridder(ds_in, ds_out, method='bilinear')  # conservative  # conservative_normed

ds_out = regridder(ds_in)
for key in ds_in.data_vars:
    if 'units' in ds_in[key].attrs:
        ds_out[key].attrs['units'] = ds_in[key].units
        ds_out[key].attrs['long_name'] = ds_in[key].long_name

# add coordinates
ds_out['latitude'] = wrf_grid_ds.XLAT_M
ds_out['longitude'] = wrf_grid_ds.XLONG_M
ds_out = ds_out.set_coords(['latitude', 'longitude'])

encoding = {  # fix issue with saving time
    'time': {
        'units': 'seconds since 2010-01-01',
        'dtype': 'float64'  # Use float instead of int64
    }
}
ds_out.to_netcdf(path=args.tropomi_out, mode='w', encoding=encoding)
print(args.tropomi_out)
print("DONE")
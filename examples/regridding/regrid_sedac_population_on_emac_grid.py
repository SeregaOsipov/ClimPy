import numpy as np
import xarray as xr
import argparse
# this will fix the esmf import bug
import os
from pathlib import Path

from climpy.utils.grid_utils import cell_area

if 'ESMFMKFILE' not in os.environ:  # os.environ.get('READTHEDOCS') and
    # RTD doesn't activate the env, and esmpy depends on a env var set there
    # We assume the `os` package is in {ENV}/lib/pythonX.X/os.py
    # See conda-forge/esmf-feedstock#91 and readthedocs/readthedocs.org#4067
    print('fixing ESMFMKFILE env variable')
    os.environ['ESMFMKFILE'] = str(Path(os.__file__).parent.parent / 'esmf.mk')
#
import xesmf as xe


__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

'''
Conservatively regrid SEDAC dataset onto EMAC grid.
This version is for Evgeniya Predybaylo for Global Methane Pledge study
'''

parser = argparse.ArgumentParser()
parser.add_argument("--mode", help="pycharm")
parser.add_argument("--host", help="pycharm")
parser.add_argument("--port", help="pycharm")
parser.add_argument("--emac_in", help="EMAC input file path")#, default='/work/mm0062/b302052/people/evgenia/EXP/CTRL/CTRL__2099__ECHAM5.nc')
parser.add_argument("--sedac_in", help="SEDAC file path")#, default='/work/mm0062/b302074/Data/NASA/SEDAC/population_density/gpw_v4_population_density_rev11_2pt5_min.nc')  # '/work/mm0062/b302074/Data/NASA/SEDAC/gpw_v4_population_count_adjusted_rev11_2pt5_min.nc')
parser.add_argument("--sedac_out", help="regridded SEDAC output file path")#, default='/work/mm0062/b302074/Data/AirQuality/AQABA/IC_BC/aux/gpw_v4_population_density_rev11_2pt5_min.nc_regrid.nc')
args = parser.parse_args()

#%% levante test case
# args.emac_in = '/work/mm0062/b302052/people/evgenia/EXP/CTRL/CTRL__2099__ECHAM5.nc'
# args.sedac_in = '/work/mm0062/b302074/Data/NASA/SEDAC/population_count/30_minutes_res/gpw_v4_population_count_rev11_30_min.nc'
# args.sedac_out = '/work/mm0062/b302074/Data/NASA/SEDAC/population_count/30_minutes_res/gpw_v4_population_count_rev11_30_min_regrid_to_emac.nc'

print('Will regrid this SEDAC onto this EMAC:\nin {}\nout {}'.format(args.sedac_in, args.emac_in))

#%% Input
ds_in = xr.open_dataset(args.sedac_in)  # SEDAC
pop_key = 'Population Count, v4.11 (2000, 2005, 2010, 2015, 2020): 30 arc-minutes'
ds_in = ds_in[pop_key][4]  # 2020 pop count. Check gpw_v4_netcdf_contents_rev11.csv for details on dimensions
ds_in = ds_in.fillna(0)  # fill NA values, otherwise conservation properties will break
print('Total population in: {}'.format(ds_in.sum()))
#%% Build source grid & var. Conservative regridding requires corners
ds_grid = xr.open_dataset(args.emac_in)[['lon', 'lat']]
print('Lon[0,0]: {}, Lon[-1,-1]: {}'.format(ds_grid['lon'][0].item(), ds_grid['lon'][-1].item()))  # print rho grid coordinates
print('Lat[0,0]: {}, Lat[-1,-1]: {}'.format(ds_grid['lat'][0].item(), ds_grid['lat'][-1].item()))
ds_out = ds_grid
#%% regridder conserves the flux (average over the cell). To fix this, convert to population density and then back
area_in = cell_area(ds_in.to_dataset())
area_out = cell_area(ds_out)
ds_in /= area_in
#%%
regridder = xe.Regridder(ds_in, ds_out, method='conservative_normed', periodic=True)  # bilinear  conservative conservative_normed
# regridder  # print out
ds_out = regridder(ds_in, keep_attrs=True)  # When dealing with global grids, we need to set periodic=True, otherwise data along the meridian line will be missing.
ds_out *= area_out
ds_out = ds_out.rename('pop_count_2020')
print('Total population out: {}'.format(ds_out.sum()))
ds_out.to_netcdf(path=args.sedac_out, mode='w')

print("DONE")

#%% OLD implmentation where I build output grid myself
# # corners are required for conservative remapping
# # build a c-grid from rho_grid
# dlon = np.diff(ds_grid['lon'])
# dlat = np.diff(ds_grid['lat'])
#
# lon_m = ds_grid.lon.data  # switch to numpy to avoid indecies overlap
# lat_m = ds_grid.lat.data
#
# #xlong_c = xr.concat([lon_m[0] - dlon[0]/2, (lon_m[1:]+lon_m[:-1])/2, lon_m[-1] + dlon[-1]/2], dim='lon')  # this has a duplicating coordinate
# xlong_c = np.concatenate([[lon_m[0] - dlon[0]/2,], (lon_m[1:] + lon_m[:-1]) / 2, [lon_m[-1] + dlon[-1] / 2,]])
# # xlong_c = xlong_c.rename({'lon': 'lon_c'})
# xlat_c = np.concatenate([[lat_m[0] - dlat[0] / 2,], (lat_m[1:] + lat_m[:-1]) / 2, [lat_m[-1] + dlat[-1] / 2,]])
# # xlat_c = xlat_c.rename({'lat': 'lat_c'})
#
# xlong_m = ds_grid['lon'].data
# xlat_m = ds_grid['lat'].data
#
# # have to convert 1d grid to 2d grid
# xlong_m, xlat_m = np.meshgrid(xlong_m, xlat_m)
# xlong_c, xlat_c = np.meshgrid(xlong_c, xlat_c)
# #%%
#
# ds_out = xr.Dataset({'lat': (['latitude', 'longitude'], xlat_m.data), 'lon': (['latitude', 'longitude'], xlong_m.data),
#                     'lat_b': (['south_north_stag', 'west_east_stag'], xlat_c.data), 'lon_b': (['south_north_stag', 'west_east_stag'], xlong_c.data),})  # it is important to call variables lat_b to indicate corners
# #%%
# regridder = xe.Regridder(ds_in, ds_out, method='conservative_normed', periodic=True)  # bilinear  conservative conservative_normed
# # regridder  # print out
# ds_out = regridder(ds_in, keep_attrs=True)  # When dealing with global grids, we need to set periodic=True, otherwise data along the meridian line will be missing.
# # ds_out = ds_out.rename('pop_count_2020')
# ds_out = ds_out.rename('pop_density_2020')
# ds_out.sum()
# ds_out.to_netcdf(path=args.sedac_out, mode='w')
#
# print("DONE")
import numpy as np
import xarray as xr
import xesmf as xe
import argparse
from climpy.utils.modis_utils import get_modis_montly_file_paths, get_modis_var


__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

'''
Conservatively regrid SEDAC dataset onto WRF grid 
'''

parser = argparse.ArgumentParser()
parser.add_argument("--mode", help="pycharm")
parser.add_argument("--port", help="pycharm")
parser.add_argument("--wrf_in", help="wrf input file path")#, default='/work/mm0062/b302074/Data/AirQuality/AQABA/IC_BC/geo_em.d01.nc')
parser.add_argument("--sedac_in", help="SEDAC file path")#, default='/work/mm0062/b302074/Data/NASA/SEDAC/population_density/gpw_v4_population_density_rev11_2pt5_min.nc')  # '/work/mm0062/b302074/Data/NASA/SEDAC/gpw_v4_population_count_adjusted_rev11_2pt5_min.nc')
parser.add_argument("--sedac_out", help="regridded SEDAC output file path")#, default='/work/mm0062/b302074/Data/AirQuality/AQABA/IC_BC/aux/gpw_v4_population_density_rev11_2pt5_min.nc_regrid.nc')
args = parser.parse_args()

print('Will regrid this SEDAC onto this WRF:\nin {}\nout {}'.format(args.sedac_in, args.wrf_in))

#%% Build source grid & var. Conservative regridding requires corners
ds_grid = xr.open_dataset(args.wrf_in)
print('Lon[0,0]: {}, Lon[-1,-1]: {}'.format(ds_grid['XLONG_M'][0,0,0].item(), ds_grid['XLONG_M'][0,-1,-1].item()))  # print rho grid coordinates
print('Lat[0,0]: {}, Lat[-1,-1]: {}'.format(ds_grid['XLAT_M'][0,0,0].item(), ds_grid['XLAT_M'][0,-1,-1].item()))
xlong_m = ds_grid['XLONG_M'][0]
xlat_m = ds_grid['XLAT_M'][0]
xlong_c = ds_grid['XLONG_C'][0]  # corners are required for conservative remapping
xlat_c = ds_grid['XLAT_C'][0]

ds_in = xr.open_dataset(args.sedac_in)  # SEDAC
# ds_out = xr.Dataset({'lat': (['latitude', 'longitude'], xlat_m.data), 'lon': (['latitude', 'longitude'], xlong_m.data),})
ds_out = xr.Dataset({'lat': (['latitude', 'longitude'], xlat_m.data), 'lon': (['latitude', 'longitude'], xlong_m.data),
                    'lat_b': (['south_north_stag', 'west_east_stag'], xlat_c.data), 'lon_b': (['south_north_stag', 'west_east_stag'], xlong_c.data),})  # it is important to call variables lat_b to indicate corners
#%% debug
# ds[ 'UN WPP-Adjusted Population Count, v4.11 (2000, 2005, 2010, 2015, 2020): 2.5 arc-minutes'][5]
# 4 is 2015, check gpw_v4_netcdf_contents_rev11.csv for details on dimensions
# 10 is Numeric country codes corresponding to nation-states (see lookup)

# constrain the size to speed up things
# lon_m = lon_m[np.logical_and(lon_m > xlong_c.min().mapping(), lon_m < xlong_c.max().mapping())]
# lat_m = lat_m[np.logical_and(lat_m > xlat_c.min().mapping(), lat_m < xlat_c.max().mapping())]
# ds_out = xr.Dataset({'lat': (['lat'], lat_m), 'lon': (['lon'], lon_m),})

#%%
regridder = xe.Regridder(ds_in, ds_out, method='conservative')  # bilinear  conservative_normed
# regridder  # print out
ds_out = regridder(ds_in)
ds_out.to_netcdf(path=args.sedac_out, mode='w')

print("DONE")
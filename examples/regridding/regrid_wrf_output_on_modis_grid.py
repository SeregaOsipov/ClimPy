import numpy as np
import xarray as xr
import xesmf as xe
import argparse
from climpy.utils.modis_utils import get_modis_montly_file_paths, get_modis_var

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

'''
Conservatively regrid WRF output onto MODIS grid 
'''

parser = argparse.ArgumentParser()
parser.add_argument("--mode", help="pycharm")
parser.add_argument("--port", help="pycharm")
parser.add_argument("--wrf_in", help="wrf input file path", default='/work/mm0062/b302074/Data/AirQuality/AQABA/chem_100_v22/output/pp_wrf_optics/merge/wrfout_d01_monmean')
parser.add_argument("--wrf_out", help="wrf output file path", default='/work/mm0062/b302074/Data/AirQuality/AQABA/chem_100_v22/output/pp_wrf_optics/merge/wrfout_d01_monmean_regrid')
args = parser.parse_args()

print('Will process this WRF:\nin {}\nout {}'.format(args.wrf_in, args.wrf_out))

#%% get MODIS grid
key = 'AOD_550_Dark_Target_Deep_Blue_Combined_Mean_Mean'
year = 2017
fps = get_modis_montly_file_paths(year)
modis_aod_data = np.zeros((12, 180, 360))
fp_modis = fps[0]
vo = get_modis_var(fp_modis, key)
#%% Build source grid & var. Conservative regridding requires corners
fp_grid = '/work/mm0062/b302074/Data/AirQuality/AQABA/IC_BC/geo_em.d01.nc'
ds_grid = xr.open_dataset(fp_grid)
# print rho grid coordinates for Jos
print('Lon[0,0]: {}, Lon[-1,-1]: {}'.format(ds_grid['XLONG_M'][0,0,0].item(), ds_grid['XLONG_M'][0,-1,-1].item()))
print('Lat[0,0]: {}, Lat[-1,-1]: {}'.format(ds_grid['XLAT_M'][0,0,0].item(), ds_grid['XLAT_M'][0,-1,-1].item()))

xlong_c = ds_grid['XLONG_C'][0]
xlat_c = ds_grid['XLAT_C'][0]

fp_fields = args.wrf_in  # '/work/mm0062/b302074/Data/AirQuality/AQABA/chem_100_v22/output/pp_wrf_optics/merge/wrfout_d01_monmean'
ds = xr.open_dataset(fp_fields)

lon_m = vo['lon']
lat_m = vo['lat']
# constrain the size to WRF domain to speed up things
lon_m = lon_m[np.logical_and(lon_m > xlong_c.min().mapping(), lon_m < xlong_c.max().mapping())]
lat_m = lat_m[np.logical_and(lat_m > xlat_c.min().mapping(), lat_m < xlat_c.max().mapping())]

ds_out = xr.Dataset({'lat': (['lat'], lat_m), 'lon': (['lon'], lon_m),})

regridder = xe.Regridder(ds, ds_out, 'bilinear')
# regridder  # print out
ds_out = regridder(ds)

# Filter out points outside original grid
ind = ds_out['so4aj'][0,0] == 0  # Get the mask first
keys = list(set(ds_out.variables.keys()) - set(['XTIME', 'lon', 'lat']))  # , 'PH', 'PHB']))
for key in keys:
    ds_out[key] = ds_out[key].where(np.logical_not(ind))
ds_out.to_netcdf(path=args.wrf_out, mode='w')

#%% plotting
# plt.figure()
# dr.isel(XTIME=0).isel(bottom_top=10).plot(x='XLONG', y='XLAT', vmin=0, vmax=14)
# plt.figure()
# dr_out.isel(XTIME=0).isel(bottom_top=10).plot(x='lon', y='lat', vmin=0, vmax=14)
#
# plt.figure()
# ax = plt.axes(projection=ccrs.PlateCarree())
# dr.isel(XTIME=0).isel(bottom_top=0).plot.pcolormesh(ax=ax, vmin=0, vmax=14)
# ax.coastlines()
#
# plt.figure()
# ax = plt.axes(projection=ccrs.PlateCarree())
# dr_out.isel(XTIME=0).isel(bottom_top=0).plot.pcolormesh(ax=ax, vmin=0, vmax=14)
# ax.coastlines()

# #%% Plot old & new fields
# bounds = np.arange(0, 14, 0.1)
# norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
#
# fig, axes = plt.subplots(nrows=2, ncols=1, subplot_kw={'projection': ccrs.PlateCarree()})
# ax = axes[0]
# plt.sca(ax)
# ax.coastlines()
# plt.pcolormesh(src_grid.get_coords(0), src_grid.get_coords(1), src_field.data, norm=norm, transform=ccrs.PlateCarree())
#
# ax = axes[1]
# plt.sca(ax)
# ax.coastlines()
# plt.pcolormesh(dst_grid.get_coords(0), dst_grid.get_coords(1), dst_field.data, norm=norm, transform=ccrs.PlateCarree())
#
# ax = axes[0]
# lims = [
#     np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
#     np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
# ]
# ax = axes[1]
# # ax.set_xlim(lims)
# # ax.set_ylim(lims)
#
# plt.colorbar()
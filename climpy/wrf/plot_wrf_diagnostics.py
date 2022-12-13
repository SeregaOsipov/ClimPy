import netCDF4
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import os
from climpy.util.plotting_utils import save_fig
from climpy.utils.netcdf_utils import convert_time_data_impl
from glob import glob
__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'


### The scripts needs to be generalized

pics_output_folder = os.path.expanduser('~') + '/Pictures/Papers/AirQuality/AQABA'
plt.ioff()


def plot_all_netcdf_variables(nc_file_path, pics_output_folder):
    nc = netCDF4.Dataset(nc_file_path)

    lat_key = 'XLAT'
    lon_key = 'XLONG'
    time_key = 'XTIME' # CDO style

    lat = nc.variables[lat_key][:]
    lon = nc.variables[lon_key][:]

    var_keys = list(nc.variables.keys())
    for key in (lat_key, lon_key, time_key, 'XTIME_bnds'):
        var_keys.remove(key)

    for var_key in var_keys:
        print('processing {}'.format(var_key))
        time_size = nc.variables[time_key].size
        for time_index in range(time_size):
            date = convert_time_data_impl(nc.variables[time_key][time_index], nc.variables[time_key].units)
            print('\ttime index {} out of {}: {}'.format(time_index, time_size, date))
            level_index = 0

            def get_wrf_variable(nc, var_key):
                dims = nc.variables[var_key].dimensions
                level_dim_key = 'bottom_top'
                var = None
                if level_dim_key in dims:
                    var = nc.variables[var_key][time_index, level_index]
                else:
                    var = nc.variables[var_key][time_index]

                return var

            var = get_wrf_variable(nc, var_key)

            plt.figure()
            # ar = AreaRangeVO('Domain', np.min(lat), np.max(lat), np.min(lon), np.max(lon))
            # map = Basemap(projection='mill', resolution='l', llcrnrlon=ar.lon_min, llcrnrlat=ar.lat_min, urcrnrlon=ar.lon_max, urcrnrlat=ar.lat_max)
            map = Basemap(projection='lcc', lat_1=27.962, lat_2=27.962, lat_0=27.962, lon_0=42.022,
                          resolution='l', width=451*10*10**3, height=451*10*10**3)
                          # llcrnrlon=ar.lon_min, llcrnrlat=ar.lat_min, urcrnrlon=ar.lon_max, urcrnrlat=ar.lat_max)
            map.drawcoastlines(linewidth=0.25)
            map.drawcountries(linewidth=0.25)
            # map.fillcontinents(color='coral', lake_color='aqua')
            # draw the edge of the map projection region (the projection limb)
            # map.drawmapboundary(fill_color='aqua')

            map.drawmeridians(np.arange(0, 360, 10), labels=[0, 0, 0, 1])  # , fontsize=20)
            map.drawparallels(np.arange(-90, 90, 10), labels=[1, 0, 0, 0])  # , fontsize=20)

            x, y = map(lon, lat)
            map.contourf(x, y, var)
            map.colorbar()

            plt.title('{} ({}), {},\n{}'.format(nc.variables[var_key].description, var_key, nc.variables[var_key].units, date))
            plt.tight_layout()
            save_fig(pics_output_folder + '/diags/{}/'.format(var_key), '{}.png'.format(date))

            plt.close()


# file_mask = '/work/mm0062/b302074/Data/AirQuality/AQABA/run_chem_on/monmean/wrfout_d01_monmean_*_sfc'
file_mask = '/work/mm0062/b302074/Data/AirQuality/AQABA/run_chem_on/monmean/wrfout_d01_monmean_*'
files = glob(file_mask)
for file_path in files:
    plot_all_netcdf_variables(file_path, pics_output_folder)

# nc_file_path = '/work/mm0062/b302074/Data/AirQuality/AQABA/run_chem_on/monmean/wrfout_d01_monmean_PM2_5_DRY_sfc'
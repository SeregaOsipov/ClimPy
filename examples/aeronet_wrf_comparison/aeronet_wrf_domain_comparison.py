import climpy.utils.aeronet_utils as aeronet
from shapely.geometry.polygon import Polygon
import netCDF4
import numpy as np
import climpy.utils.wrf_utils as wrf_utils
import climpy.utils.grid_utils as grid
import pandas as pd
import matplotlib.pyplot as plt
import climpy.utils.file_path_utils as fpu
from climpy.utils.plotting_utils import save_figure_bundle, save_figure, JGR_page_width_inches
from scipy.constants import golden
import argparse
import os

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

"""
This script plots WRF-Aeronet diagnostics within the WRF domain.
This version is slow. Much faster approach is to extract data into separate netcdf at Aeronet locations first (sample_wrf_output_at_aeronet_locations.py)
and then plot using aeronet_wrf_pp_comparison.py

To run it from the bash, see list of input args below.


Example (one line for copy-paste):
python aeronet_wrf_domain_comparison.py --aeronet_in='/work/mm0062/b302074//Data/NASA/Aeronet/' --wrf_in=/work/mm0062/b302074/Data/AirQuality/AQABA/chem_106/output/wrfout_d01_2017-0*_00:00:00 --diags_out=/work/mm0062/b302074//Pictures//Papers/AirQuality/AQABA/chem_106/ --aod_level=15

The same example (split for readability):
python aeronet_wrf_domain_comparison.py
        --aeronet_in='/work/mm0062/b302074//Data/NASA/Aeronet/'
        --wrf_in=/work/mm0062/b302074/Data/AirQuality/AQABA/chem_106/output/wrfout_d01_2017-0*_00:00:00
        --diags_out=/work/mm0062/b302074//Pictures//Papers/AirQuality/AQABA/chem_106/
        --aod_level=15
"""

parser = argparse.ArgumentParser()
parser.add_argument("--aeronet_in", help="Aeronet file path, which contains AOD and INV folders (download all)", required=True)
parser.add_argument("--wrf_in", help="WRF file path, for example, /storage/.../wrfout_d01_2017-*_00:00:00", required=True)
parser.add_argument("--diags_out", help="Figures/diags file path, for example, /storage/.../Pictures/Paper/", required=True)
parser.add_argument("--aod_level", help="Aeronet AOD level", default=15)
# have to add those to make script Pycharm compatible
parser.add_argument("--mode", help="pycharm")
parser.add_argument("--port", help="pycharm")
# TODO: add time range as input
args = parser.parse_args()

# this is my defaults for debugging
aeronet.DATA_FILE_PATH_ROOT = fpu.get_aeronet_file_path_root()
wrf_file_path = fpu.get_root_storage_path_on_hpc() + '/Data/AirQuality/AQABA/chem_106/output/wrfout_d01_2017-*_00:00:00'
pics_output_folder = fpu.get_pictures_root_folder() + '/Papers/AirQuality/AQABA/{}/'.format('chem_106')
aod_level = 15

# parse the user input and override defaults
if 'PYCHARM_HOSTED' not in os.environ.keys():
    print('Using the provided args')
    aeronet.DATA_FILE_PATH_ROOT = args.aeronet_in
    wrf_file_path = args.wrf_in
    pics_output_folder = args.diags_out
    aod_level = args.aod_level

print('STARTING DIAGS for \n wrf_in {} \n Aeronet v{} in {} \n output diags into {}'.format(wrf_file_path, aod_level, aeronet.DATA_FILE_PATH_ROOT, pics_output_folder))

# Preparations are done, start diags

# get WRF grid
nc = netCDF4.MFDataset(wrf_file_path)  # or netcdf4.Dataset
lon = nc['XLONG'][0]  # my grid is stationary in time
lat = nc['XLAT'][0]  # sample it at time index 0
domain = Polygon([(np.min(lon), np.min(lat)), (np.min(lon), np.max(lat)), (np.max(lon), np.max(lat)), (np.max(lon), np.min(lat))])
wrf_time = wrf_utils.generate_netcdf_uniform_time_data(nc.variables['Times'])
time_range = (wrf_time.min(), wrf_time.max())

stations = aeronet.filter_available_stations(domain, time_range, aod_level)

# plot the map with stations
# WRF utils do not support MFDataset, just get any file
nc_first = netCDF4.Dataset(nc._files[0])
fig, ax = wrf_utils.plot_domain(nc_first)
import cartopy.crs as crs
ax.plot(stations['Longitude(decimal_degrees)'], stations['Latitude(decimal_degrees)'],
         color='orange', marker='o', linewidth=0,
         transform=crs.PlateCarree(),)
save_figure_bundle(pics_output_folder + '/aeronet/', 'WRF domain and Aeronet stations maps')


def get_wrf_aod_at_aeronet_location(station):
    yx_tuple, lon_p, lat_p, distance_error = grid.find_closest_grid_point(lon, lat,
                                                                          station['Longitude(decimal_degrees)'],
                                                                          station['Latitude(decimal_degrees)'])
    # check that station is not too far, i.e. distance_error < some value, say 0.1
    if distance_error > 1:
        print('station {} is too far'.format(station['Site_Name']))
        return None
    # read OD profile (t, z, y, x)
    AOD = nc.variables['TAUAER3'][:, :, yx_tuple[0], yx_tuple[1]]
    column_AOD = np.sum(AOD, axis=1)

    vo = {}
    vo['time'] = wrf_time
    vo['data'] = column_AOD

    return vo


aeronet_var_key = 'AOD_500nm'
wrf_var_key = 'AOD_600nm'

# now lets process all stations, prepare data first

wrf_list = []
aeronet_list = []
for index, station in stations.iterrows():
    print('Processing {}'.format(station['Site_Name']))
    aeronet_vo = aeronet.get_aod_diag('*{}*'.format(station['Site_Name']), aeronet_var_key, aod_level, aeronet.ALL_POINTS,
                                      time_range=time_range)
    # Aeronet data availability is monthly, time filter can still yield no data coverage
    if aeronet_vo['data'].size == 0:
        print('No data for Aeronet station {}, skipping it'.format(station['Site_Name']))
        continue

    wrf_vo = get_wrf_aod_at_aeronet_location(station)
    # is station is too far wrf will return None
    if wrf_vo is None:
        print('No data from WRF for Aeronet station {}, skipping it'.format(station['Site_Name']))
        continue

    wrf_list.append(wrf_vo)
    aeronet_list.append(aeronet_vo)

print('Data prepared, do plotting')
print('Do time series plot')

# Time series plot for each site individually
for wrf_vo, aeronet_vo, dummy in zip(wrf_list, aeronet_list, stations.iterrows()):
    station = dummy[1]
    fig = plt.figure(constrained_layout=True, figsize=(JGR_page_width_inches(), JGR_page_width_inches() / golden))
    plt.plot(wrf_vo['time'], wrf_vo['data'], 'o', label='WRF, {}'.format(wrf_var_key))
    plt.plot(aeronet_vo['time'], aeronet_vo['data'], '*', label='Aeronet v{}, {}'.format(aod_level, aeronet_var_key))
    plt.ylabel('Optical depth, ()')
    plt.xlabel('Time, ()')
    plt.legend()
    plt.title('Column AOD at {}'.format(station['Site_Name']))
    #save_figure_bundle(pics_output_folder + '/aeronet/', 'WRF-Aeronet AOD, {}'.format(station['Site_Name']))
    save_figure(pics_output_folder + '/aeronet/by_site/', 'WRF-Aeronet AOD, {}.svg'.format(station['Site_Name']))
    plt.close(fig)

print('Do scatter plot')

# Next, do the scatter plot
plt.figure(constrained_layout=True, figsize=(JGR_page_width_inches(), JGR_page_width_inches()))
plt.grid()
plt.axis('equal')

for wrf_vo, aeronet_vo, dummy in zip(wrf_list, aeronet_list, stations.iterrows()):
    station = dummy[1]
    # Do the scatter plot, bring data to the same resolution
    df = pd.DataFrame(data=wrf_vo['data'], index=wrf_vo['time'], columns=[wrf_var_key])
    wrf_df = df.resample('D').mean()
    df = pd.DataFrame(data=aeronet_vo['data'], index=aeronet_vo['time'], columns=[aeronet_var_key])
    aeronet_df = df.resample('D').mean()

    # make sure that values are compared at the same time
    d1, model_ind, obs_ind = np.intersect1d(wrf_df.index, aeronet_df.index, return_indices=True)
    plt.scatter(aeronet_df.iloc[obs_ind], wrf_df.iloc[model_ind], label=station['Site_Name'])

plt.legend()
# y=x line
#global_max = np.max([wrf_df.iloc[model_ind].max(), aeronet_df.iloc[obs_ind].max()])
global_max = 3
plt.plot([0, global_max], [0, global_max], 'k-')

plt.xlabel('Observations')
plt.ylabel('Model')
plt.title('Aeronet v{} {}, WRF {}'.format(aod_level, aeronet_var_key, wrf_var_key))

save_figure_bundle(pics_output_folder + '/aeronet/', 'WRF-Aeronet AOD scatter, all stations')

print('DONE')



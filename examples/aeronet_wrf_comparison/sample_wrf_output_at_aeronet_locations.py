import climpy.utils.aeronet_utils as aeronet

import subprocess
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
from climpy.utils.wrf_chem_utils import CHEM_100_AEROSOLS_KEYS
from climpy.utils.file_path_utils import convert_file_path_mask_to_list

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

'''
This script will extract the variables for comparison at Aeronet locations
This is purely to speed up the comparison later due to netcdf specifics

This version runs in parallel among all files but for each station, because python can not run on multiple nodes 
'''

# this is my defaults for debugging
sim_version = 'chem_100_v4'
is_wrf_in_required = 'PYCHARM_HOSTED' not in os.environ.keys()

parser = argparse.ArgumentParser()
parser.add_argument("--aeronet_in", help="Aeronet file path, which contains AOD and INV folders (download all)",
                    required=False, default=fpu.get_aeronet_file_path_root())
parser.add_argument("--wrf_in", help="WRF file path, for example, /storage/.../wrfout_d01_2017-*_00:00:00",
                    default=fpu.get_root_storage_path_on_hpc() + '/Data/AirQuality/AQABA/{}/output/wrfout_d01_2017-*_00:00:00'.format(sim_version),
                    required=is_wrf_in_required)
parser.add_argument("--aod_level", help="Aeronet AOD level", default=15)
# have to add those to make script Pycharm compatible
parser.add_argument("--mode", help="pycharm")
parser.add_argument("--port", help="pycharm")
# TODO: add time range as input

args = parser.parse_args()
aeronet.DATA_FILE_PATH_ROOT = args.aeronet_in
wrf_file_path = args.wrf_in
# pics_output_folder = args.diags_out
aod_level = args.aod_level


print('STARTING DIAGS for \n wrf_in {} \n Aeronet v{} in {} \n'.format(wrf_file_path, aod_level, aeronet.DATA_FILE_PATH_ROOT))


# get WRF grid
nc = netCDF4.MFDataset(wrf_file_path)  # or netcdf4.Dataset
lon = nc['XLONG'][0]  # my grid is stationary in time
lat = nc['XLAT'][0]  # sample it at time index 0
domain = Polygon([(np.min(lon), np.min(lat)), (np.min(lon), np.max(lat)), (np.max(lon), np.max(lat)), (np.max(lon), np.min(lat))])
wrf_time = wrf_utils.generate_netcdf_uniform_time_data(nc.variables['Times'])
time_range = (wrf_time.min(), wrf_time.max())

stations = aeronet.filter_available_stations(domain, time_range, aod_level)

wrf_dir = os.path.dirname(wrf_file_path)
print('Processing WRF out in {}'.format(wrf_dir))

station = stations.iloc[0]
for index, station in stations.iterrows():
    print('\n\tProcessing Aeronet station {}/{}: {}\n'.format(index, len(stations), station['Site_Name']))

    # get wrf files
    wrf_file_path = '{}/wrfout_d01_2017-*_00:00:00'.format(wrf_dir)
    wrf_in_paths = convert_file_path_mask_to_list(wrf_file_path)

    processes = []
    wrf_out_paths = []
    for file_index, wrf_in_path in zip(range(len(wrf_in_paths)), wrf_in_paths):
        print('\tProcessing WRF input {}/{}: {}'.format(file_index, len(wrf_in_paths), wrf_in_path))

        wrf_out_path = '{}/pp_aeronet/{}_{}'.format(wrf_dir, os.path.basename(wrf_in_path), station['Site_Name'])
        wrf_out_paths.append(wrf_out_path)

        # ncks -v PH,PHB,nu0,ac0,corn,NU3,AC3,COR3,TAUAER3 -d XLONG,34.7822 -d XLAT,-30.855 infile.nc outfile.nc
        # cdo -P 16 remapnn,lon=39.1047/lat=22.3095 -select,name=TAUAER3,PH,PHB,nu0,ac0,corn,NU3,AC3,COR3 wrfout_d01_2017-*_00:00:00 ./pp_aeronet/wrfout_d01_

        # .run is blocking
        p = subprocess.Popen(['cdo', '-P', '1', 'remapnn,lon={}/lat={}'.format(station['Longitude(decimal_degrees)'], station['Latitude(decimal_degrees)']),
                        '-select,name=TAUAER3,TAUAER4,ALT,PH,PHB,nu0,ac0,corn,NU3,AC3,COR3,{}'.format(','.join(CHEM_100_AEROSOLS_KEYS)),
                        wrf_in_path,
                        wrf_out_path])
        processes.append(p)

    print('cdo remapnn submitted, waiting to finish')
    for p in processes:
        p.communicate()

    # merge in time
    base_name = os.path.basename(wrf_out_paths[0])
    merged_wrf_out_path = '{}/pp_aeronet/{}_{}'.format(wrf_dir, base_name[0:10], station['Site_Name'])
    p = subprocess.Popen(['cdo', '-P', '8', 'mergetime', ' '.join(wrf_out_paths), merged_wrf_out_path])
    p.communicate()

    # remove temp files
    for path in wrf_out_paths:
        print('Removing {}'.format(path))
        os.remove(path)

print('DONE')


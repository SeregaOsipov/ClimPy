from climpy.utils.file_path_utils import get_root_storage_path_on_hpc
import netCDF4
from climpy.utils.diag_decorators import time_interval_selection
# from climpy.utils.time_utils import process_time_range_impl
from climpy.utils.netcdf_utils import generate_netcdf_uniform_time_data
import numpy as np

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'


@time_interval_selection
def prepare_avhrr_aod(zonal_mean=False):
    file_name = 'aot_avhrr_1989-1992'
    lat_name = 'latitude'
    lon_name = 'longitude'

    if zonal_mean:
        file_name += '_zonal_mean'
        lat_name = 'lat'
        lon_name = 'lon'

    file_path = get_root_storage_path_on_hpc() + 'Data/AVHRR/AOT/' + file_name + '.nc'
    nc = netCDF4.Dataset(file_path)

    time_data = generate_netcdf_uniform_time_data(nc.variables['time'])
    # t_slice, time_data = process_time_range_impl(time_data[:], time_range_vo)

    aod_data = nc.variables['aot1'][:]  # [t_slice]
    if zonal_mean:
        aod_data = np.squeeze(aod_data)

    lat_data = nc.variables[lat_name][:]
    lon_data = nc.variables[lon_name][:]

    vo = {}
    vo['data'] = aod_data
    vo['time'] = time_data
    vo['lat'] = lat_data
    vo['lon'] = lon_data

    return vo

import netCDF4
import argparse

from climpy.wrf.WPS_netcdf_ungrib.wps_unnetcdf_utils import prepare_nc_data, wrf_write, _FIELD_MAP_ECMWF_OA_2_WRF, \
    GAUSSIAN_PROJECTION

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'





# //TODO: REDO THE LOGIC SIMILAR TO MERRA2

# //TODO: check the projection variables




# The script is designed for the ECMWF Operational Analysis
# See http://www2.mmm.ucar.edu/wrf/users/docs/user_guide/users_guide_chap3.html  # _Writing_Meteorological_Data

parser = argparse.ArgumentParser()
parser.add_argument("nc_file_path", help="unnetcdf similar to the WPS ungrib, provide file path")
args = parser.parse_args()

print('supplied file path is: ' + args.nc_file_path)

nc_file_path = args.nc_file_path

# this map holds nc name to field conversion, see Vtable
_FIELD_MAP = _FIELD_MAP_ECMWF_OA_2_WRF

# 2d fields
# nc_file_path = '/shaheen/project/k1090/osipovs/Data/temp_download/ECMWF-OA/netcdf/global/F1280/sfc/ECMWF-OA_sfc_20160101_20160101.nc'
# 3d fields
# nc_file_path = '/shaheen/project/k1090/osipovs/Data/temp_download/ECMWF-OA/netcdf/global/F1280/ml/ECMWF-OA_ml_20160101_00.nc'
# nc_file_path = '/shaheen/project/k1090/osipovs/Data/temp_download/ECMWF-OA/netcdf/global/F1280/ml/ECMWF-OA_ml_20160101_06.nc'
nc = netCDF4.Dataset(nc_file_path)


var_list = nc.variables.keys()
var_list = list(var_list)
# drop longitude latitude and time
var_list.remove('latitude')
var_list.remove('longitude')
var_list.remove('time')

n_vert_levels = 1
is_model_levels = False
#deduce is it a surface or model levels file
if 'level' in nc.dimensions.keys():
    #3d
    is_model_levels = True
    var_list.remove('level')
    n_vert_levels = nc.dimensions['level'].size
else:
    var_list.remove('rsn') #auxiliary variable

print('list of variables to process: ' + str(var_list))


for time_index in range(nc.variables['time'].shape[0]):
    print('Processing time index {}'.format(time_index))
    # open file for writing
    f = None
    for var_name in var_list:
        print('processing variable '+ var_name)
        for level_index in range(n_vert_levels):
            level_type_postfix = 'sfc'
            if ( is_model_levels ):
                level_type_postfix = 'ml'
                print('processing level {}'.format(level_index))

            nc_data = prepare_nc_data(nc, _FIELD_MAP, var_name, time_index, level_index, GAUSSIAN_PROJECTION)
            out_file_name = 'FILE_unnc_' + level_type_postfix + ':' + nc_data['time_data'].strftime('%Y-%m-%d_%H')
            if ( f is None ):
                print('Creating file ' + out_file_name)
                f = open(out_file_name, 'wb')

            wrf_write(f, nc_data)

    print('Closing file ' + out_file_name)
    f.close()

    print("startlat: {}, deltalat: {}".format(nc_data['startlat'], nc_data['deltalat']))
    print("startlon: {}, deltalon: {}".format(nc_data['startlon'], nc_data['deltalon']))

print('DONE')

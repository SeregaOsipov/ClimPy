import argparse
import xarray as xr
import netCDF4
import numpy as np
import datetime as dt
from dateutil import rrule
import os
from climpy.wrf.WPS_netcdf_ungrib.wps_unnetcdf_utils import prepare_nc_data, wrf_write, _FIELD_MAP_MERRA_2_WRF, \
    get_merra2_file_path, LATLON_PROJECTION, format_hdate

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

'''
This script extracts the data from the MERRA2 reAnalysis and outputs it into the WPS intermediate format (for WRF)
See http://www2.mmm.ucar.edu/wrf/users/docs/user_guide/users_guide_chap3.html  # _Writing_Meteorological_Data
The script also performs several auxiliary steps:
1. precomputes the 3d Pressure and thus calc_ecmwf_p.exe step is unnecessary
2. DOES NOT Interpolate the soil temperature and humidity profiles onto the WPS vertical grid
3. YOU NEED TO INCLUDE NEW SOIL VARIABLES INTO THE METGRID.TBL
   METGRID.TBL.ARW.MERRA2 is provided as an example
4. Soil moisture is implemented very sloppy due to poor MERRA2 output availability
5. SST is probably replaced by the Skin Temperature

These are some of the MERRA2 docs for quick reference
1. https://gmao.gsfc.nasa.gov/pubs/docs/Bosilovich785.pdf
2. https://gmao.gsfc.nasa.gov/pubs/docs/Reichle541.pdf

Run command example:
python -u ${CLIMPY}/climpy/wrf/WPS_netcdf_ungrib/wps_unnetcdf_merra2.py --start_date=2022-10-25 --end_date=2023-01-01 >& log.unnetcdf_aread

TODO: replace MERRA2 predownloading with the OpenDAP access
'''


parser = argparse.ArgumentParser()
parser.add_argument("--start_date", help="unnetcdf is similar to the WPS ungrib, provide the start and end dates in the YYYY-MM-DD format")
parser.add_argument("--end_date", help="unnetcdf is similar to the WPS ungrib, provide the end date in the YYYY-MM-DD format")
parser.add_argument("--mode", "--port", help="the are only to support pycharm debugging")
args = parser.parse_args()

out_storage_path = '/home/osipovs/workspace/WRF/Data/unnetcdf/'
out_storage_path = '/project/k1090/osipovs/Data/NASA/MERRA2/unnetcdf/'
out_storage_path = '/work/mm0062/b302074/Data/NASA/MERRA2/unnnetcdf/'
# out_storage_path = '/Users/osipovs2/Temp/'  # local debug
try:
    os.makedirs(out_storage_path)
except FileExistsError:
    print('probably unnetcdf storage directory already exists')


start_date = dt.datetime(2017, 1, 1)
end_date = dt.datetime(2017, 1, 2)
if args.start_date:
    start_date = dt.datetime.strptime(args.start_date, '%Y-%m-%d')
if args.end_date:
    end_date = dt.datetime.strptime(args.end_date, '%Y-%m-%d')

print('start and end dates are: {} and {}'.format(start_date, end_date))

# given the start, end dates and interval, generate the list of exact time stamps to be processed

requested_dates = list(rrule.rrule(rrule.HOURLY, interval=3, dtstart=start_date, until=end_date))
requested_dates = requested_dates[:-1]


# this map holds nc name to field conversion, see Vtable
_FIELD_MAP = _FIELD_MAP_MERRA_2_WRF


# MERRA stores data in separate datasets/files
# build the list of the datasets and variables to extract from them
dataset_2d = {"name": "inst1_2d_asm_Nx", "vars": list(_FIELD_MAP.keys())[0:7]}
dataset_2d_ocn = {"name": "tavg1_2d_ocn_Nx", "vars": list(_FIELD_MAP.keys())[7:8]}
dataset_2d_const = {"name": "const_2d_asm_Nx", "vars": list(_FIELD_MAP.keys())[8:10]}
# dataset_2d_land = {"name": "tavg1_2d_lnd_Nx", "vars": list(_FIELD_MAP.keys())[10:19]}
# dataset_3d = {"name": "inst3_3d_asm_Nv", "vars": list(_FIELD_MAP.keys())[19:25]}
dataset_2d_land = {"name": "tavg1_2d_lnd_Nx", "vars": list(_FIELD_MAP.keys())[10:22]}
dataset_3d = {"name": "inst3_3d_asm_Nv", "vars": list(_FIELD_MAP.keys())[22:28]}
# list of the all the datasets to process
datasets = (dataset_2d, dataset_2d_ocn, dataset_2d_const, dataset_2d_land, dataset_3d)
datasets = (dataset_2d_const, dataset_2d, dataset_2d_ocn, dataset_2d_land, dataset_3d)

for requested_date in requested_dates:
    print('Processing date {}'.format(requested_date))

    # create output file_paths: sfc & ml. Existing files will be destroyed
    # if you want nondestructive logic - implement it here
    out_file_name_sfc = out_storage_path + '/FILE_unnc_' + 'sfc' + ':' + requested_date.strftime('%Y-%m-%d_%H')
    print('Creating file ' + out_file_name_sfc)
    f_sfc = open(out_file_name_sfc, 'wb')

    out_file_name_ml = out_storage_path + '/FILE_unnc_' + 'ml' + ':' + requested_date.strftime('%Y-%m-%d_%H')
    print('Creating file ' + out_file_name_ml)
    f_ml = open(out_file_name_ml, 'wb')

    for dataset in datasets:
        var_list = dataset['vars']
        print('DATA SET is {} and the list of variables to process is {}'.format(dataset['name'], dataset['vars']))

        nc_file_path, is_df_time_invariant = get_merra2_file_path(dataset['name'], requested_date)

        time_dependent_df = xr.open_dataset(nc_file_path)
        if is_df_time_invariant:  # time invariant data sets should only have one element in time dimension
            df = time_dependent_df.isel(time=0)
        else:  # Ocean data are 30 minutes shifted relative to atmosphere. Allow 30 minutes tolerance
            df = time_dependent_df.sel(time=requested_date, drop=True, method='nearest', tolerance=np.timedelta64(30,'m'))  # do the time selection.
            df['time'] = requested_date  # have to store the date somewhere

        # deduce is it a surface or model levels file
        n_vert_levels = 1
        f = f_sfc
        if _FIELD_MAP['level'] in df.dims.keys():
            n_vert_levels = df.dims[_FIELD_MAP['level']]
            f = f_ml

        for var_key in var_list:
            print('processing variable {}'.format(var_key))
            for level_index in range(n_vert_levels):
                print('processing level {}'.format(level_index))
                nc_data = prepare_nc_data(df, _FIELD_MAP, var_key, level_index, LATLON_PROJECTION)

                if _FIELD_MAP[var_key].override_time:
                    # in general the date has to be identical, but constant fields may prescribe it for the wrong date
                    nc_data['time_data'] = requested_date
                    nc_data['hdate'] = format_hdate(requested_date)
                    print('date was overridden for the var_key {}'.format(var_key))

                wrf_write(f, nc_data)

                # print("startlat: {}, deltalat: {}".format(nc_data['startlat'], nc_data['deltalat']))
                # print("startlon: {}, deltalon: {}".format(nc_data['startlon'], nc_data['deltalon']))

    print('Closing file ' + out_file_name_sfc)
    f_sfc.close()
    print('Closing file ' + out_file_name_ml)
    f_ml.close()

print('DONE')

import argparse
import xarray as xr
import netCDF4
import numpy as np
import datetime as dt
from dateutil import rrule
import os
from climpy.wrf.WPS_netcdf_ungrib.wps_unnetcdf_utils import prepare_nc_data, wrf_write, _FIELD_MAP_MERRA_2_WRF, \
    get_merra2_file_path, LATLON_PROJECTION, format_hdate
import pandas as pd

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
python -u ${CLIMPY}/climpy/wrf/WPS_netcdf_ungrib/wps_unnetcdf_merra2.py --start_date=2017-01-01 --end_date=2017-06-15 >& log.unnetcdf_2017_jan-jun
# On Compute Node
python -u ${CLIMPY}/climpy/wrf/WPS_netcdf_ungrib/wps_unnetcdf_merra2.py --start_date=2023-05-15 --end_date=2023-06-01 --out_storage_path=/scratch/osipovs/Data/NASA/MERRA2/unnetcdf/ >& log.unnetcdf_2023_05

TODO: replace MERRA2 predownloading with the OpenDAP access

To process by month in parallel, use wps_unnetcdf_merra2_in_parallel.sh

'''

parser = argparse.ArgumentParser()
parser.add_argument("--mode", "--port", help="the are only to support pycharm debugging")
parser.add_argument("--start_date", help="2023-05-15")
parser.add_argument("--end_date", help="2023-06-01")
parser.add_argument("--MERRA2_STORAGE_PATH", help="where MERRA2 files are", default='/project/k10048/Data/NASA/MERRA2/')
parser.add_argument("--out_storage_path", help="where to put unnetcdf files", default='/project/k10048/Data/NASA/MERRA2/unnetcdf/')  # /scratch/osipovs/Data/NASA/MERRA2/unnetcdf/
args = parser.parse_args()

# args.out_storage_path = '/scratch/osipovs/Data/NASA/MERRA2/unnetcdf/'
# args.start_date = '2023-05-15'
# args.end_date = '2023-06-01'

print('start and end dates are: {} and {}'.format(args.start_date, args.end_date))
os.makedirs(args.out_storage_path, exist_ok=True)
requested_dates = pd.period_range(start=args.start_date, end=args.end_date, freq='3h')
_FIELD_MAP = _FIELD_MAP_MERRA_2_WRF  # this map holds nc name to field conversion, see Vtable

# MERRA stores data in separate datasets/files
# build the list of the datasets and variables to extract from them
dataset_2d = {"name": "inst1_2d_asm_Nx", "vars": list(_FIELD_MAP.keys())[0:7]}
dataset_2d_ocn = {"name": "tavg1_2d_ocn_Nx", "vars": list(_FIELD_MAP.keys())[7:8]}
dataset_2d_const = {"name": "const_2d_asm_Nx", "vars": list(_FIELD_MAP.keys())[8:10]}
# dataset_2d_land = {"name": "tavg1_2d_lnd_Nx", "vars": list(_FIELD_MAP.keys())[10:19]}
# dataset_3d = {"name": "inst3_3d_asm_Nv", "vars": list(_FIELD_MAP.keys())[19:25]}
dataset_2d_land = {"name": "tavg1_2d_lnd_Nx", "vars": list(_FIELD_MAP.keys())[10:22]}
dataset_3d = {"name": "inst3_3d_asm_Nv", "vars": list(_FIELD_MAP.keys())[22:28]}

# list of all datasets to process
datasets = (dataset_2d, dataset_2d_ocn, dataset_2d_const, dataset_2d_land, dataset_3d)
datasets = (dataset_2d_const, dataset_2d, dataset_2d_ocn, dataset_2d_land, dataset_3d)

for raw_date in requested_dates:
    requested_date = raw_date.to_timestamp()
    requested_date_str = requested_date.strftime('%Y-%m-%d_%H')
    print('Processing date {}'.format(requested_date))

    # create output file_paths: sfc & ml. Existing files will be destroyed
    # if you want nondestructive logic - implement it here
    out_file_name_sfc = args.out_storage_path + '/FILE_unnc_' + 'sfc' + ':' + requested_date_str
    print('Creating file ' + out_file_name_sfc)
    f_sfc = open(out_file_name_sfc, 'wb')

    out_file_name_ml = args.out_storage_path + '/FILE_unnc_' + 'ml' + ':' + requested_date_str
    print('Creating file ' + out_file_name_ml)
    f_ml = open(out_file_name_ml, 'wb')

    for dataset in datasets:
        var_list = dataset['vars']
        print('DATA SET is {} and the list of variables to process is {}'.format(dataset['name'], dataset['vars']))

        nc_file_path, is_df_time_invariant = get_merra2_file_path(dataset['name'], requested_date, args.MERRA2_STORAGE_PATH)

        time_dependent_ds = xr.open_dataset(nc_file_path)
        if is_df_time_invariant:  # time invariant data sets should only have one element in time dimension
            ds = time_dependent_ds.isel(time=0)
        else:  # Ocean data are 30 minutes shifted relative to atmosphere. Allow 30 minutes tolerance
            ds = time_dependent_ds.sel(time=requested_date, drop=True, method='nearest', tolerance=np.timedelta64(30, 'm'))  # do the time selection.
            ds['time'] = requested_date  # have to store the date somewhere

        # deduce is it a surface or model levels file
        n_vert_levels = 1
        f = f_sfc
        if _FIELD_MAP['level'] in ds.sizes.keys():
            n_vert_levels = ds.sizes[_FIELD_MAP['level']]
            f = f_ml

        for var_key in var_list:
            print('processing variable {}'.format(var_key))
            for level_index in range(n_vert_levels):
                print('processing level {}'.format(level_index))
                nc_data = prepare_nc_data(ds, _FIELD_MAP, var_key, level_index, LATLON_PROJECTION)

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

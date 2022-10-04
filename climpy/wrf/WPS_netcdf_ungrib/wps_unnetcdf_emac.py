import argparse
import netCDF4
import numpy as np
import datetime as dt
from dateutil import rrule
import os
from climpy.wrf.WPS_netcdf_ungrib.wps_unnetcdf_utils import prepare_nc_data, wrf_write, \
    get_merra2_file_path, LATLON_PROJECTION, format_hdate, _FIELD_MAP_EMAC_2_WRF, get_emac_file_path
from climpy.utils.netcdf_utils import convert_time_data_impl


__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'


'''
unnetcdf is similar to the WPS ungrib.

This script extracts the data from the EMAC/MESSY model run and outputs it into the WPS intermediate format (for WRF)
See http://www2.mmm.ucar.edu/wrf/users/docs/user_guide/users_guide_chap3.html  # _Writing_Meteorological_Data
The script also performs several auxiliary steps:
1. precomputes the 3d Pressure and thus calc_ecmwf_p.exe step is unnecessary
2. Set ups the soil temperature and humidity profiles on the model grid. METGRID.TBL needs to be updated accordingly
3. YOU NEED TO INCLUDE NEW SOIL VARIABLES INTO THE METGRID.TBL
   METGRID.TBL.ARW.MERRA2 is provided as an example
4. Soil moisture is implemented very sloppy due to poor land model in EMAC
5. SST is probably replaced by the Skin Temperature


This is how to run script in the terminal:
gogomamba
python -u ${CLIMPY}/climpy/wrf/WPS_netcdf_ungrib/wps_unnetcdf_emac.py

2050 example:
gogomamba
python -u ${CLIMPY}/climpy/wrf/WPS_netcdf_ungrib/wps_unnetcdf_emac.py --emac_in=/work/mm0062/b302074/Data/AirQuality/EMME/2050/IC_BC/emac --out=/work/mm0062/b302074/Data/AirQuality/EMME/2050/IC_BC/unnetcdf/ --start_date=2050-01-01_00 --end_date=2050-01-02_00

Ignore or update current default arguments  
'''


parser = argparse.ArgumentParser()
parser.add_argument("--start_date", help="start date in the YYYY-MM-DD_HH format", default='2017-06-15_00')
parser.add_argument("--end_date", help="end date in the YYYY-MM-DD_HH format", default='2017-09-02_00')
parser.add_argument("--emac_in", help="folder containing emac output")  #, default='/work/mm0062/b302011/script/Osipov/simulations/AQABA')
parser.add_argument("--out", help="folder to store unnc")  # , default='/work/mm0062/b302074/Data/AirQuality/EMME/IC_BC/2050/unnetcdf/')
parser.add_argument("--emac_sim_label", help="folder containing emac output", default='MIM_STD________')   # sim label has fixed width and then filled with ___
parser.add_argument("--mode", "--port", help="the are only to support pycharm debugging")
args = parser.parse_args()

#%%
out_storage_path = args.out

try:
    os.makedirs(out_storage_path)
except FileExistsError:
    print('probably unnetcdf storage directory already exists')

# if args.start_date:
start_date = dt.datetime.strptime(args.start_date, '%Y-%m-%d_%H')
end_date = dt.datetime.strptime(args.end_date, '%Y-%m-%d_%H')

print('start and end dates are: {} and {}'.format(start_date, end_date))
print('emac_in and out are: {} and {}'.format(args.emac_in, args.out))

# generate the list of exact time stamps to be processed
requested_dates = list(rrule.rrule(rrule.HOURLY, interval=3, dtstart=start_date, until=end_date))
requested_dates = requested_dates[:-1]

# this map holds nc name to field conversion, see Vtable
_FIELD_MAP = _FIELD_MAP_EMAC_2_WRF

# build the list of the datasets and variables to extract from them
dataset_echam = {"name": "ECHAM5", "vars": list(_FIELD_MAP.keys())[0:6]}
dataset_e5vdiff = {"name": "e5vdiff", "vars": list(_FIELD_MAP.keys())[6:7]}
dataset_g3b = {"name": "g3b", "vars": list(_FIELD_MAP.keys())[7:25]}

# list of the all the datasets to process
datasets = (dataset_echam, dataset_e5vdiff, dataset_g3b)

# subset for debugging
# dataset_g3b = {"name": "g3b", "vars": list(_FIELD_MAP.keys())[23:25]}
# datasets = (dataset_g3b, )

get_file_path_impl = get_emac_file_path  # get_merra2_file_path
# multifile_support = True  # swap to False, if netcdfs are not compatible
use_multifile_support = False
multifile_support_on_daily_output = True  # EMAC makes restart and creates new file with hour snapshot

for requested_date in requested_dates:
    print('Processing date {}'.format(requested_date))

    # create output file_paths: sfc & ml
    # they will be destroyed, if the already exist
    # if you need nondestructive logic - do it here
    out_file_name_sfc = out_storage_path + '/FILE_unnc_' + 'sfc' + ':' + requested_date.strftime('%Y-%m-%d_%H')
    print('Creating file ' + out_file_name_sfc)
    f_sfc = open(out_file_name_sfc, 'wb')

    out_file_name_ml = out_storage_path + '/FILE_unnc_' + 'ml' + ':' + requested_date.strftime('%Y-%m-%d_%H')
    print('Creating file ' + out_file_name_ml)
    f_ml = open(out_file_name_ml, 'wb')

    for dataset in datasets:
        var_list = dataset['vars']
        print('DATA SET is {} and the list of variables to process is {}'.format(dataset['name'], dataset['vars']))

        nc_file_path = get_file_path_impl(args.emac_in, args.emac_sim_label, dataset['name'], requested_date, use_multifile_support, multifile_support_on_daily_output)
        if use_multifile_support or multifile_support_on_daily_output:
            nc = netCDF4.MFDataset(nc_file_path)
        else:
            nc = netCDF4.Dataset(nc_file_path)


        def get_time_index_in_netcdf(nc):
            # check if it is the time invariant dataset (constants)
            # TODO: check
            # if 'Time-invariant' in nc.LongName:
            #     return 0

            nc_dates = convert_time_data_impl(nc['time'], nc['time'].units)

            generate_dates = False
            if generate_dates:  # instead generate time and use the fact that all file_paths are daily
                # reading the actual time stamps causes issues, since we have instantaneous and time average fields
                # nc_start_date = dt.datetime.strptime(nc.RangeBeginningDate, '%Y-%m-%d')
                nc_start_date = dt.datetime(nc_dates[0].year, nc_dates[0].month, nc_dates[0].day, nc_dates[0].hour, nc_dates[0].minute, 0)
                nc_end_date = dt.datetime(nc_dates[0].year, nc_dates[0].month+1, nc_dates[0].day, nc_dates[0].hour, nc_dates[0].minute, 0)
                # nc_end_date = nc_start_date + dt.timedelta(days=1)
                # hours_interval = int(24/nc.dimensions['time'].size)  # TODO: FIX ME
                hours_interval = 10
                nc_dates = list(rrule.rrule(rrule.HOURLY, interval=hours_interval, dtstart=nc_start_date, until=nc_end_date))
                nc_dates = np.array(nc_dates)
                nc_dates = nc_dates[:-1]

            # find the requested date in the file
            time_index = np.where(nc_dates == requested_date)[0]
            if time_index.size == 0:
                raise Exception('unnetcdf_MERRA2', 'requested time {} can not be found in the netcdf file {}'.format(requested_date, nc_file_path))
            time_index = time_index[0]

            return time_index


        time_index = get_time_index_in_netcdf(nc)

        for var_key in var_list:
            print('processing variable {}'.format(var_key))
            nc_var_key = _FIELD_MAP[var_key].netcdf_var_key  # key in EMAC

            # deduce is it a surface or model levels file
            n_vert_levels = 1
            f = f_sfc
            if _FIELD_MAP['level'] in nc.variables[nc_var_key].dimensions:
                n_vert_levels = nc.dimensions[_FIELD_MAP['level']].size
                f = f_ml

            for level_index in range(n_vert_levels):
                print('processing level {}'.format(level_index))
                nc_data = prepare_nc_data(nc, _FIELD_MAP, var_key, time_index, level_index, LATLON_PROJECTION)

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

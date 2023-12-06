import argparse
from distutils.util import strtobool

import numpy as np
import xarray as xr
import datetime as dt
from dateutil import rrule
import os

from climpy.utils.debug_utils import detailed_print
from climpy.wrf.WPS_netcdf_ungrib.wps_unnetcdf_utils import prepare_nc_data, wrf_write, \
    get_merra2_file_path, LATLON_PROJECTION, format_hdate, _FIELD_MAP_EMAC_2_WRF, get_emac_file_path, WrfMetgridMapItem
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
   METGRID.TBL.ARW.AQABA is provided as an example
4. Soil moisture is implemented very sloppy due to poor land model in EMAC
5. SST is probably replaced by the Skin Temperature


How to run script in the terminal:

EMME 2050 example:
gogomamba
year=2050
scenario=CLE  
python -u ${CLIMPY}/climpy/wrf/WPS_netcdf_ungrib/wps_unnetcdf_emac.py --emac_in=/work/mm0062/b302074/Data/AirQuality/EMME/${year}/${scenario}/IC_BC/emac/ --out=/work/mm0062/b302074/Data/AirQuality/EMME/${year}/${scenario}/IC_BC/unnetcdf/ --start_date=${year}-01-01_00 --end_date=$((year+1))-01-02_00 --emac_sim_label=test01_________ >& log.unnetcdf_${year}_${scenario}

EMME 2017 example:
gogomamba
year=2017
scenario=
python -u ${CLIMPY}/climpy/wrf/WPS_netcdf_ungrib/wps_unnetcdf_emac.py --emac_in=/work/mm0062/b302074/Data/AirQuality/EMME/${year}/${scenario}/IC_BC/emac/ --out=/work/mm0062/b302074/Data/AirQuality/EMME/${year}/${scenario}/IC_BC/unnetcdf/ --start_date=${year}-01-01_00 --end_date=$((year+1))-01-02_00 --emac_sim_label=test01_________ >& log.unnetcdf_${year}_${scenario}

Ignore or update current default arguments  
'''

#%%

parser = argparse.ArgumentParser()
parser.add_argument("--start_date", help="start date in the YYYY-MM-DD_HH format", default='2017-06-15_00')
parser.add_argument("--end_date", help="end date in the YYYY-MM-DD_HH format", default='2017-09-02_00')
parser.add_argument("--emac_in", help="folder containing emac output")  #, default='/work/mm0062/b302011/script/Osipov/simulations/AQABA')
parser.add_argument("--out", help="folder to store unnc")  # , default='/work/mm0062/b302074/Data/AirQuality/EMME/IC_BC/2050/unnetcdf/')
parser.add_argument("--emac_sim_label", help="folder containing emac output", default='MIM_STD________')   # sim label has fixed width and then filled with ___
parser.add_argument("--detailed_debug", help="True/False", type=strtobool, default=False)
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
# dataset_echam = {"name": "ECHAM5", "vars": list(_FIELD_MAP.keys())[0:5]}
# dataset_e5vdiff = {"name": "e5vdiff", "vars": list(_FIELD_MAP.keys())[5:7]}
# dataset_g3b = {"name": "g3b", "vars": list(_FIELD_MAP.keys())[7:25]}

var_list = list(_FIELD_MAP.keys())[0:25]
# streams = ('ECHAM5', 'e5vdiff', 'g3b', 'WRF_bc')  # old format where EMAC groups output into its own streams
streams = ('WRF_bc_met', 'WRF_bc_chem',)  # new format, where Andrea extracted everything related to WRF into dedicated streams

# list of the all the datasets to process
# datasets = (dataset_echam, dataset_e5vdiff, dataset_g3b)

get_file_path_impl = get_emac_file_path  # get_merra2_file_path
# multifile_support = True  # swap to False, if netcdfs are not compatible
use_multifile_support = False  # TODO: currently, xarray merging across different streams will not work (probably, because it has to merge and concatenate at the same time)
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

    # employ xarray to merge across different EMAC streams into single DataSet
    nc_stream_file_paths = ()
    xr_streams = ()
    for stream in streams:  # collect several EMAC output streams into single xarray
        nc_stream_file_path = get_file_path_impl(args.emac_in, args.emac_sim_label, stream, requested_date, use_multifile_support, multifile_support_on_daily_output)
        nc_stream_file_paths += (nc_stream_file_path,)
        print(nc_stream_file_path)
        xr_stream = xr.open_mfdataset(nc_stream_file_path)
        xr_streams += (xr_stream,)

    time_dependent_df = xr.merge(xr_streams)

    print('Will open and merge the following streams: {}'.format(nc_stream_file_paths))
    df = time_dependent_df.sel(time=requested_date, drop=True)  # do the time selection
    df['time'] = requested_date  # have to store the date somewhere

    print('The list of variables to process is {}'.format(var_list))

    for var_key in var_list:
        if isinstance(_FIELD_MAP[var_key], WrfMetgridMapItem):
            print('processing variable {}'.format(var_key))
        else:
            print('Skipping {} var as auxiliary'.format(var_key))
            continue

        nc_var_key = _FIELD_MAP[var_key].netcdf_var_key  # key in EMAC

        # deduce is it a surface or model levels file
        n_vert_levels = 1
        f = f_sfc
        if _FIELD_MAP['level'] in df.variables[nc_var_key].dims:
            n_vert_levels = df.dims[_FIELD_MAP['level']]
            f = f_ml

        for level_index in range(n_vert_levels):
            detailed_print('processing level {}'.format(level_index), args.detailed_debug)
            nc_data = prepare_nc_data(df, _FIELD_MAP, var_key, level_index, LATLON_PROJECTION)

            if _FIELD_MAP[var_key].override_time:
                # in general the date has to be identical, but constant fields may prescribe it for the wrong date
                nc_data['time_data'] = requested_date
                nc_data['hdate'] = format_hdate(requested_date)
                print('date was overridden for the var_key {}'.format(var_key))

            wrf_write(f, nc_data)

    print('Closing file ' + out_file_name_sfc)
    f_sfc.close()
    print('Closing file ' + out_file_name_ml)
    f_ml.close()

print('DONE')

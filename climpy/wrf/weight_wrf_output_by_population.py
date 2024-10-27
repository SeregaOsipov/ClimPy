import netCDF4
import os
import pandas as pd
import numpy as np
import xarray as xr
import wrf as wrf
from climpy.utils.atmos_utils import compute_column_from_vmr_profile
import argparse
from distutils.util import strtobool
from climpy.utils.pop_utils import compute_pop_weighted_diags_by_country, get_present_population

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

'''
Run examples:
python -u ${CLIMPY}climpy/wrf/weight_wrf_output_by_population.py --wrf_in=/work/mm0062/b302074/Data/AirQuality/EMME/2050/HLT/chem_100_v1/output/pp_health/wrfout_d01_timmean --wrf_out=/work/mm0062/b302074/Data/AirQuality/EMME/2050/HLT/chem_100_v1/output/pp_health/wrfout_d01_timmean_pop_wtd --excel_out=/work/mm0062/b302074/Data/AirQuality/EMME/2050/HLT/chem_100_v1/output/pp_health/wrfout_d01_timmean_pop_wtd.xls

rootPath=/work/mm0062/b302074/  # LEVANTE
rootPath=/scratch/osipovs/  # Shaheen

# health
for scenario in 2017 2050/HLT 2050/CLE 2050/MFR 2050/MFR_LV
do
    echo $scenario
    # hourly
    python -u ${CLIMPY}climpy/wrf/weight_wrf_output_by_population.py --wrf_in=${rootPath}/Data/AirQuality/EMME/${scenario}/chem_100_v1/output/pp_health/wrfout_d01_*_00 --wrf_out=${rootPath}/Data/AirQuality/EMME/${scenario}/chem_100_v1/output/pp_health/wrfout_d01_hourly_pop_wtd &
    # timmean
    # python -u ${CLIMPY}climpy/wrf/weight_wrf_output_by_population.py --wrf_in=${rootPath}/Data/AirQuality/EMME/${scenario}/chem_100_v1/output/pp_health/wrfout_d01_timmean --wrf_out=${rootPath}/Data/AirQuality/EMME/${scenario}/chem_100_v1/output/pp_health/wrfout_d01_timmean_pop_wtd --excel_out=${rootPath}/Data/AirQuality/EMME/${scenario}/chem_100_v1/output/pp_health/wrfout_d01_timmean_pop_wtd.xls
done
wait

# climate
for scenario in 2017 2050/HLT 2050/CLE 2050/MFR 2050/MFR_LV
do
    echo $scenario
    # hourly
    python -u ${CLIMPY}climpy/wrf/weight_wrf_output_by_population.py --var_keys=twb --wrf_in=${rootPath}/Data/AirQuality/EMME/${scenario}/chem_100_v1/output/pp_wrf/wrfout_d01_*_00 --wrf_out=${rootPath}/Data/AirQuality/EMME/${scenario}/chem_100_v1/output/pp_wrf/wrfout_d01_hourly_pop_wtd &
    # timmean
    # python -u ${CLIMPY}climpy/wrf/weight_wrf_output_by_population.py --var_keys=twb --wrf_in=${rootPath}/Data/AirQuality/EMME/${scenario}/chem_100_v1/output/pp_wrf/wrfout_d01_timmean --wrf_out=${rootPath}/Data/AirQuality/EMME/${scenario}/chem_100_v1/output/pp_wrf/wrfout_d01_timmean_pop_wtd --excel_out=${rootPath}/Data/AirQuality/EMME/${scenario}/chem_100_v1/output/pp_wrf/wrfout_d01_timmean_pop_wtd.xls
    # python -u ${CLIMPY}climpy/wrf/weight_wrf_output_by_population.py --var_keys=T2,Q2 --wrf_in=${rootPath}/Data/AirQuality/EMME/${scenario}/chem_100_v1/output/cdo/wrfout_d01_timmean --wrf_out=${rootPath}/Data/AirQuality/EMME/${scenario}/chem_100_v1/output/cdo/wrfout_d01_timmean_pop_wtd --excel_out=${rootPath}/Data/AirQuality/EMME/${scenario}/chem_100_v1/output/cdo/wrfout_d01_timmean_pop_wtd.xls
done
wait


# COAWST, twb (pp_wrf)
rootPath=/scratch/osipovs/  # Shaheen
for scenario in CLE HLT MFR MFR_LV present_day/aer_on present_day/aer_off
do
    echo $scenario
    simPath=${rootPath}/Data/COAWST/EMME/2017/${scenario}/
    
    # hourly
    #python -u ${CLIMPY}/climpy/wrf/weight_wrf_output_by_population.py --var_keys=twb --wrf_in=${simPath}/chem_100_v1/output/pp_wrf/wrfout_d01_*_00 --wrf_out=${simPath}/chem_100_v1/output/pp_wrf/wrfout_d01_hourly_pop_wtd &
    python -u ${CLIMPY}/climpy/wrf/weight_wrf_output_by_population.py --var_keys=T2,Q2 --wrf_in=${simPath}/chem_100_v1/output/wrfout_d01_*_00 --wrf_out=${simPath}/chem_100_v1/output/pop_wtd/wrfout_d01_hourly_pop_wtd &
    
    # timmean
    # python -u ${CLIMPY}/climpy/wrf/weight_wrf_output_by_population.py --var_keys=twb --wrf_in=${simPath}/chem_100_v1/output/pp_wrf/wrfout_d01_timmean --wrf_out=${simPath}/chem_100_v1/output/pp_wrf/wrfout_d01_timmean_pop_wtd --excel_out=${simPath}/chem_100_v1/output/pp_wrf/wrfout_d01_timmean_pop_wtd.xls
    # python -u ${CLIMPY}/climpy/wrf/weight_wrf_output_by_population.py --var_keys=T2,Q2 --wrf_in=${simPath}/chem_100_v1/output/cdo/wrfout_d01_timmean --wrf_out=${simPath}/chem_100_v1/output/cdo/wrfout_d01_timmean_pop_wtd --excel_out=${simPath}/chem_100_v1/output/cdo/wrfout_d01_timmean_pop_wtd.xls
done
wait
'''


parser = argparse.ArgumentParser()
parser.add_argument("--mode", "--port", "--host", help="pycharm")
parser.add_argument("--wrf_in", help="wrf input file path")#, default='/work/mm0062/b302074/Data/AirQuality/EMME/2050/EDGAR_trend/chem_100_v2/output/pp_wrf/wrfout_d01_2050-12-09_00_00_00')
parser.add_argument("--wrf_out", help="wrf output file path")#, default='/work/mm0062/b302074/Data/AirQuality/EMME/2050/EDGAR_trend/chem_100_v2/output/pp_wrf/pop_wtd/wrfout_d01_2050-12-09_00_00_00')
parser.add_argument("--excel_out", help="excel output file path", default=None)#, default='/work/mm0062/b302074/Data/AirQuality/EMME/2050/EDGAR_trend/chem_100_v2/output/pp_wrf/pop_wtd/wrfout_d01_2050-12-09_00_00_00')
parser.add_argument("--var_keys", help="comma separated list of vars to weight, e.g.: twb,t2", default=None)
parser.add_argument("--select_boa", help="True/False", type=strtobool, default=False)
args = parser.parse_args()

#debug
# import climpy.utils.file_path_utils as fpu
# fpu.set_env('workstation')  # configure file paths to local machine
# args.wrf_in = '/HDD2/Data/AirQuality/EMME/2050/CLE/chem_100_v1/output/pp_health/wrfout_d01_timmean'
# args.wrf_out = '/HDD2/Data/AirQuality/EMME/2050/CLE/chem_100_v1/output/pp_health/wrfout_d01_timmean_pop_wtd'
# args.excel_out = '/HDD2/Data/AirQuality/EMME/2050/CLE/chem_100_v1/output/pp_health/wrfout_d01_timmean_pop_wtd.xls'
# hourly
# args.wrf_in='/work/mm0062/b302074/Data/AirQuality/EMME/2050/HLT/chem_100_v1/output/wrfout_d01_2050-01-13_00_00_00'
# args.wrf_out='/work/mm0062/b302074/Data/AirQuality/EMME/2050/HLT/chem_100_v1/output/pop_wtd/wrfout_d01_2050-01-13_00_00_00'
# args.var_keys='T2,Q2,o3,no,no2'
# args.select_boa=True

# args.wrf_in='/scratch/osipovs/Data/COAWST/EMME/2017/HLT/chem_100_v1/output/pp_wrf/wrfout_d01_2017-01-13_00_00_00'
# args.wrf_out='/scratch/osipovs/Data/COAWST/EMME/2017/HLT/chem_100_v1/output/pp_wrf/wrfout_d01_2017-01-13_00_00_00_pop_wtd'
# args.var_keys='twb'  # T2,Q2,o3,no,no2'

print('Will process WRF files:\nin {}\nout {}'.format(args.wrf_in, args.wrf_out))
print('Will keep this WRF vars:\n{}'.format(args.var_keys))
#%% Parallel attempt
# import glob
# from natsort import natsorted
# files = glob.glob(args.wrf_in)
# files = natsorted(files)
# xr_in = xr.open_mfdataset(files, combine='nested', concat_dim='Time', parallel=True)#, engine='netcdf4')

# client = Client(threads_per_worker=1).  # setup 1 thread per worked if running netcdf in parallel
# args.wrf_in='/work/mm0062/b302074/Data/AirQuality/EMME/2050/MFR/chem_100_v1/output/pp_health/wrfout_d01_2050-0*_00'
# xr_in = xr.open_mfdataset(args.wrf_in, combine='nested', concat_dim='Time', parallel=True)#, engine='netcdf4')
#%%
xr_in = xr.open_mfdataset(args.wrf_in, combine='nested', concat_dim='Time')
# xr_in = xr.open_dataset(args.wrf_in)

xr_in = xr_in.drop_vars('XTIME_bnds', errors='ignore')
if 'Time' in xr_in.coords:
    xr_in = xr_in.drop_vars('XTIME')  # if netcdf was processed by xarray, then I may already have a Time variable
else:
    xr_in = xr_in.rename({'XTIME':'Time'})

if args.var_keys is not None:
    print('Selecting the following vars {}'.format(args.var_keys))
    xr_in = xr_in[args.var_keys.split(',')]

if args.select_boa:
    print('Selecting only BOA:\n{}'.format(args.select_boa))
    xr_in = xr_in.isel(bottom_top=0)

preload_xarray_dataset = True
if preload_xarray_dataset:  # preload xr_in to speed up the calculations
    print('preloading the dataset')
    xr_in.load()
    print('done loading xr_in')


population_ds = get_present_population()
print('starting weightening')
ds_weighted_by_country = compute_pop_weighted_diags_by_country(xr_in, population_ds)
print('done weightening')

fix_the_dim_order = True
if fix_the_dim_order:  # Time needs to be first dim, but country was added at axis=0 (to be CDO complaint)
    # dims = list(ds_weighted_by_country.dims)
    # reordered_dims = dims[1:] + [dims[0],]
    reordered_dims = ds_weighted_by_country.dims
    print('Reorder dimensions to this order: {}'.format(reordered_dims))
    ds_weighted_by_country = ds_weighted_by_country.transpose(*reordered_dims)

if not os.path.exists(os.path.dirname(args.wrf_out)):
    os.makedirs(os.path.dirname(args.wrf_out))
ds_weighted_by_country.to_netcdf(args.wrf_out)

if args.excel_out is not None:
    with pd.ExcelWriter(args.excel_out, engine="openpyxl") as writer:
        ds_weighted_by_country.to_dataframe().to_excel(writer, sheet_name="Sheet1")
        if 'PM1' in ds_weighted_by_country.variables:
            ds_weighted_by_country[['PM1', 'PM25', 'PM10']].to_dataframe().to_excel(writer, sheet_name="PMs")
            ds_weighted_by_country[['PM1_by_type', 'PM25_by_type', 'PM10_by_type']].to_dataframe().to_excel(writer, sheet_name="PMsByType")

print('DONE')
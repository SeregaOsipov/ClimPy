import netCDF4
import os
import numpy as np
import xarray as xr
import wrf as wrf
from climpy.utils.atmos_utils import compute_column_from_vmr_profile
import argparse
from climpy.utils.pop_utils import compute_pop_weighted_diags_by_country, get_present_population

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

'''
Script derives several common diagnostics from WRF output, such as SO2 & O3 columns in DU

--wrf_in=/work/mm0062/b302074/Data/AirQuality/EMME/2050/EDGAR_trend/chem_100_v2/output/pp_wrf/wrfout_d01_timmean --wrf_out=/work/mm0062/b302074/Data/AirQuality/EMME/2050/EDGAR_trend/chem_100_v2/output/pp_wrf/wrfout_d01_timmean_pop_wtd
'''

parser = argparse.ArgumentParser()
parser.add_argument("--mode", "--port", "--host", help="pycharm")
parser.add_argument("--wrf_in", help="wrf input file path", default='/work/mm0062/b302074/Data/AirQuality/EMME/2050/EDGAR_trend/chem_100_v2/output/pp_wrf/wrfout_d01_2050-12-09_00_00_00')  # default='/Users/osipovs/Data/AirQuality/EMME/2050/chem_100_v2/output/pp_wrf/wrfout_d01_timmean')
parser.add_argument("--wrf_out", help="wrf output file path", default='/work/mm0062/b302074/Data/AirQuality/EMME/2050/EDGAR_trend/chem_100_v2/output/pp_wrf/pop_wtd/wrfout_d01_2050-12-09_00_00_00')  # default='/Users/osipovs/Data/AirQuality/EMME/2050/chem_100_v2/output/pp_wrf/wrfout_d01_timmean_pop_wtd')
parser.add_argument("--var_keys", help="comma separated list of vars to weight, e.g.: twb,t2", default='twb')
args = parser.parse_args()

print('Will process WRF files:\nin {}\nout {}'.format(args.wrf_in, args.wrf_out))
print('Will keep this WRF vars:\n{}'.format(args.var_keys))
#%%
xr_in = xr.open_dataset(args.wrf_in, engine='netcdf4')
if args.var_keys is not None:
    xr_in = xr_in[args.var_keys.split(',')]

population_present_ds = get_present_population()
ds_weighted_by_country = compute_pop_weighted_diags_by_country(xr_in, population_present_ds)

if not os.path.exists(os.path.dirname(args.wrf_out)):
    os.makedirs(os.path.dirname(args.wrf_out))
ds_weighted_by_country.to_netcdf(args.wrf_out)

print('DONE')
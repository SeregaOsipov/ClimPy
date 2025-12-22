# %load_ext autoreload
# %autoreload 2

from climpy.utils.atmos_utils import DRY_AIR_MOLAR_MASS
import netCDF4
import os
import numpy as np
import xarray as xr
import wrf as wrf
import argparse
from climpy.utils.tropomi_utils import TROPOMI_in_WRF_KEYS, derive_tropomi_ch4_pressure_grid
from climpy.utils.wrf_utils import compute_stag_pressure, compute_stag_z, compute_dz, calculate_air_mass_dry, compute_stag_pressure_impl, compute_p, compute_stag_p, average_wrf_diag_between_tropomi_staggered_pressure_grid, generate_xarray_uniform_time_data
from wrf import Constants
import datetime as dt

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

'''
Script derives TROPOMI-specific diagnostics to enable WRF-Chem-TROPOMI comparison.

# Individual run
wrf_in=/scratch/osipovs/Data/AirQuality/THOFA/inversion/v5/run_srs_ref/wrfout_d01_2023-06-10_00_00_00
wrf_out=/scratch/osipovs/Data/AirQuality/THOFA/inversion/v5/run_srs_ref/pp/tropomi_like_ch4/wrfout_d01_2023-06-10_00_00_00
tropomi_in=/project/k10048/osipovs/Data/Copernicus/Sentinel-5P/d02/S5P_OFFL_L2__CH4____20230610T084541_20230610T102711_29311_03_020500_20230612T004748.nc
python -u ${CLIMPY}climpy/wrf/pp_wrf_tropomi_like_ch4.py --wrf_in=${wrf_in} --wrf_out=${wrf_out} --tropomi_in=${tropomi_in}

# sbatch $BASH_SCRIPTS/pp_wrf_column_average_ensemble.sh /scratch/osipovs/Data/AirQuality/THOFA/inversion/v5/run_srs_revised/wrfout_d01_2023-06-01_00_00_00 /scratch/osipovs/Data/AirQuality/THOFA/inversion/v5/run_srs_revised/pp/column/wrfout_d01_2023-06-01_00_00_00

'''
#%%
parser = argparse.ArgumentParser()
parser.add_argument("--mode", "--port", "--host", help="pycharm")
parser.add_argument("--wrf_in", help="wrf input file path")#, default='/work/mm0062/b302074/Data/AirQuality/EMME/2017/chem_100_v1/output/wrfout_d01_2017-12-14_00_00_00')
parser.add_argument("--wrf_out", help="wrf output file path")# , default='/work/mm0062/b302074/Data/AirQuality/EMME/2017/chem_100_v1/output/pp_wrf/wrfout_d01_2017-12-14_00_00_00')
parser.add_argument("--tropomi_in", help="File path to TROPOMI L2 orbit")#, default='/work/mm0062/b302074/Data/AirQuality/EMME/2017/chem_100_v1/output/wrfout_d01_2017-12-14_00_00_00')
args = parser.parse_args()
#%%
# args.wrf_in = '/scratch/osipovs/Data/AirQuality/THOFA/inversion/v4/run_srs_ref/wrfout_d01_2023-06-10_00_00_00'
# args.wrf_out = '/scratch/osipovs/Data/AirQuality/THOFA/inversion/v4/run_srs_ref/pp/tropomi_like_ch4/wrfout_d01_2023-06-10_00_00_00'
# args.tropomi_in = '/project/k10048/osipovs/Data/Copernicus/Sentinel-5P/d02/S5P_OFFL_L2__CH4____20230610T084541_20230610T102711_29311_03_020500_20230612T004748.nc'
#
# #d01
# args.wrf_in = '/scratch/osipovs/Data/AirQuality/THOFA/chem_100_v2025.0/wrfout_d01_2023-06-01_00_00_00'
# args.wrf_out = '/scratch/osipovs/Data/AirQuality/THOFA/chem_100_v2025.0/pp/tropomi_like_ch4/wrfout_d01_2023-06-01_00_00_00'
# args.tropomi_in = '/project/k10048/osipovs/Data/Copernicus/Sentinel-5P/d01/S5P_OFFL_L2__CH4____20230603T073550_20230603T091720_29211_03_020500_20230604T235242.nc'
#
# #d02
# args.wrf_in = '/scratch/osipovs/Data/AirQuality/THOFA/inversion/v5/run_srs_ref/wrfout_d01_2023-06-03_00_00_00'
# args.wrf_out = '/scratch/osipovs/Data/AirQuality/THOFA/inversion/v5/run_srs_ref/pp/tropomi_like_ch4/wrfout_d01_2023-06-03_07_35_50'
# args.tropomi_in = '/project/k10048/osipovs//Data/Copernicus/Sentinel-5P//d02/S5P_OFFL_L2__CH4____20230603T073550_20230603T091720_29211_03_020500_20230604T235242.nc'

print('Will process this WRF:\nin {}\nout {}\ntropomi {}'.format(args.wrf_in, args.wrf_out, args.tropomi_in))
#%% Prep WRF
wrf_ds = xr.open_dataset(args.wrf_in)
if 'XTIME' in wrf_ds.dims:
    wrf_ds = wrf_ds.rename({'XTIME':'Time'}).rename({'Time':'time'})
else:
    wrf_ds['time'] = generate_xarray_uniform_time_data(wrf_ds.Times)
    wrf_ds = wrf_ds.rename({'Time':'time'})
#%% Prep TROPOMI
tropomi_ds = xr.open_dataset(args.tropomi_in)
derive_tropomi_ch4_pressure_grid(tropomi_ds)
#%% Minimize the WRF ds size and interpolate in time
keys = ['PH', 'PHB', 'P', 'PB', 'PSFC', 'ZNW', 'MUB', 'MU'] + ['ch4', 'no2']
wrf_ds = wrf_ds[keys]
# tropomi_ds['time'] = dt.datetime.fromisoformat('2023-06-01T14:25:00')  # TODO: Fix tropomi time
wrf_ds = wrf_ds.interp(time=tropomi_ds.time, method='linear', kwargs={'bounds_error': True})
#%% Deriving intermediate diagnostics
compute_dz(wrf_ds)
compute_p(wrf_ds)
compute_stag_p(wrf_ds)
calculate_air_mass_dry(wrf_ds)

derive_tropomi_ch4_pressure_grid(tropomi_ds)
#%% Derive xch4 TROPOMI-like diagnostic
# See Eq 3 in https://sentinels.copernicus.eu/documents/247904/2474726/Sentinel-5P-Level-2-Product-User-Manual-Methane.pdf#page=19.59
da = average_wrf_diag_between_tropomi_staggered_pressure_grid(wrf_ds, 'ch4', tropomi_ds)
wrf_ds['xch4'] = da  # get wrf ch4 mixing ratio averaged between tropomi levels

da = average_wrf_diag_between_tropomi_staggered_pressure_grid(wrf_ds, 'air_mass_dry', tropomi_ds)
wrf_ds['dvair'] = da / DRY_AIR_MOLAR_MASS  # mol/m2 = kg / m^2 / (kg mol-1)  # dry air column

wrf_ds['dvch4'] = 10**-6 * wrf_ds['xch4'] * wrf_ds['dvair']  # mol/m2 of methane
wrf_ds['vch4'] = tropomi_ds.methane_profile_apriori.sum(dim='layer') + (tropomi_ds.column_averaging_kernel*(wrf_ds['dvch4'] - tropomi_ds.methane_profile_apriori)).sum(dim='layer')
wrf_ds['xch4_like_tropomi'] = 10**6 * wrf_ds['vch4'] / wrf_ds['dvair'].sum('layer')  # ppmv. This is like-for-like diagnostic
wrf_ds.xch4_like_tropomi.attrs['units'] = 'ppmv'
#%% Save the output
print('Saving to:\n{}'.format(args.wrf_out))
os.makedirs(os.path.dirname(args.wrf_out), exist_ok=True)

export_keys = ['xch4_like_tropomi', ]
wrf_ds[export_keys].to_netcdf(args.wrf_out)#, mode=mode, unlimited_dims=unlimited_dim, format='NETCDF4_CLASSIC')
print('Done')
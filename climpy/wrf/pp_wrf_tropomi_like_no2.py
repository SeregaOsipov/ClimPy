from climpy.utils.atmos_utils import DRY_AIR_MOLAR_MASS
%load_ext autoreload
%autoreload 2

from scipy import interpolate
import netCDF4
import os
import numpy as np
import xarray as xr
import wrf as wrf
import argparse
from climpy.utils.tropomi_utils import TROPOMI_in_WRF_KEYS, derive_tropomi_ch4_pressure_grid, derive_tropomi_no2_pressure_grid
from climpy.utils.wrf_utils import compute_stag_pressure, compute_stag_z, compute_dz, calculate_air_mass_dry, compute_stag_pressure_impl, compute_p, compute_stag_p, average_wrf_diag_between_tropomi_staggered_pressure_grid, interpolate_wrf_diag_to_tropomi_rho_pressure_grid, generate_netcdf_uniform_time_data, generate_xarray_uniform_time_data
from wrf import Constants
import datetime as dt

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'


'''
Script derives TROPOMI-specific diagnostics to enable WRF-Chem-TROPOMI comparison.

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
# d01
args.wrf_in = '/scratch/osipovs/Data/AirQuality/THOFA/chem_100_v2025.0/wrfout_d01_2023-06-01_00_00_00'
args.wrf_out = '/scratch/osipovs/Data/AirQuality/THOFA/chem_100_v2025.0/pp/tropomi_like_no2/wrfout_d01_2023-06-01_00_00_00'
args.tropomi_in = '/project/k10048/osipovs/Data/Copernicus/Sentinel-5P/d01/S5P_OFFL_L2__NO2____20230601T081351_20230601T095521_29183_03_020500_20230603T044537.nc'

#d02
args.wrf_in = '/scratch/osipovs/Data/AirQuality/THOFA/inversion/v5/run_srs_ref/wrfout_d01_2023-06-10_00_00_00'
args.wrf_out = '/scratch/osipovs/Data/AirQuality/THOFA/inversion/v5/run_srs_ref/pp/tropomi_like_no2/wrfout_d01_2023-06-10_00_00_00'
args.tropomi_in = '/project/k10048/osipovs/Data/Copernicus/Sentinel-5P/d02/S5P_OFFL_L2__NO2____20230610T084541_20230610T102711_29311_03_020500_20230612T004757.nc'

print('Will process this WRF:\nin {}\nout {}'.format(args.wrf_in, args.wrf_out))
#%% Prep WRF
wrf_ds = xr.open_dataset(args.wrf_in)
if 'XTIME' in wrf_ds.dims:
    wrf_ds = wrf_ds.rename({'XTIME':'Time'}).rename({'Time':'time'})
else:
    wrf_ds['time'] = generate_xarray_uniform_time_data(wrf_ds.Times)
    wrf_ds = wrf_ds.rename({'Time':'time'})
#%% Prep TROPOMI
tropomi_ds = xr.open_dataset(args.tropomi_in)
derive_tropomi_no2_pressure_grid(tropomi_ds)
#%% Minimize the WRF ds size and interpolate in time
# tropomi_ds['time'] = dt.datetime.fromisoformat('2023-06-01T14:25:00')  # hardcore time
keys = ['PH', 'PHB', 'P', 'PB', 'PSFC', 'ZNW', 'MUB', 'MU'] + ['ch4', 'no2']
wrf_ds = wrf_ds[keys]
wrf_ds = wrf_ds.interp(time=tropomi_ds.time, method='linear')
#%% Deriving intermediate diagnostics
compute_dz(wrf_ds)
compute_p(wrf_ds)
compute_stag_p(wrf_ds)
calculate_air_mass_dry(wrf_ds)

derive_tropomi_no2_pressure_grid(tropomi_ds)
#%%
print('Remember that interpolated NO2 profile will contain NaNs if TROPOMI top is above WRF top')
da = interpolate_wrf_diag_to_tropomi_rho_pressure_grid(wrf_ds, 'air_mass_dry', tropomi_ds)
wrf_ds['dvair'] = da / DRY_AIR_MOLAR_MASS  # mol/m2 = kg / m^2 / (kg mol-1)  # dry air column
wrf_ds['xno2'] = interpolate_wrf_diag_to_tropomi_rho_pressure_grid(wrf_ds, 'no2', tropomi_ds)  # ppmv or 10**6*mol/mol

trop_layer_index_da = tropomi_ds.tm5_tropopause_layer_index.where(tropomi_ds.qa_value>0).where((tropomi_ds.tm5_tropopause_layer_index>0) & (tropomi_ds.tm5_tropopause_layer_index<tropomi_ds.layer.size))
trop_mask_da = wrf_ds.layer <= trop_layer_index_da
wrf_ds['trop_no2_column_like_tropomi'] = (10**-6 * wrf_ds['xno2'] * wrf_ds['dvair']  * tropomi_ds.averaging_kernel).where(trop_mask_da).sum(dim='layer')  # mol/m2 of no2
wrf_ds.trop_no2_column_like_tropomi.attrs['long_name'] = 'TROPOMI-like tropospheric NO2, derived from WRF output'
wrf_ds.trop_no2_column_like_tropomi.attrs['units'] = 'mol/m2'
#%% Save the output
print('Saving to:\n{}'.format(args.wrf_out))
os.makedirs(os.path.dirname(args.wrf_out), exist_ok=True)

export_keys = ['trop_no2_column_like_tropomi', ]
wrf_ds[export_keys].to_netcdf(args.wrf_out)#, mode=mode, unlimited_dims=unlimited_dim, format='NETCDF4_CLASSIC')
print('Done')
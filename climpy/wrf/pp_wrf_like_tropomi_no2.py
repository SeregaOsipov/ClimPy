# %load_ext autoreload
# %autoreload 2

import argparse
import os

import xarray as xr

from climpy.utils.atmos_utils import DRY_AIR_MOLAR_MASS
from climpy.utils.tropomi_utils import derive_tropomi_no2_pressure_grid
from climpy.utils.wrf_utils import compute_dz, calculate_air_mass_dry, compute_p, compute_stag_p, interpolate_wrf_diag_to_tropomi_rho_pressure_grid, generate_xarray_uniform_time_data, fix_time_variable_in_wrf_output

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'


'''
Script derives TROPOMI-specific diagnostics to enable WRF-Chem-TROPOMI comparison.

TROPOMI User Guide: https://sentinels.copernicus.eu/documents/247904/2474726/Sentinel-5P-Level-2-Product-User-Manual-Nitrogen-Dioxide.pdf#page=24.09

# sbatch $BASH_SCRIPTS/pp_wrf_column_average_ensemble.sh /scratch/osipovs/Data/AirQuality/THOFA/inversion/v5/run_srs_revised/wrfout_d01_2023-06-01_00_00_00 /scratch/osipovs/Data/AirQuality/THOFA/inversion/v5/run_srs_revised/pp/column/wrfout_d01_2023-06-01_00_00_00

'''


def pp_wrf_like_tropomi_no2(args):
    print('pp_wrf_like_tropomi_no2')
    print(f'--wrf_in={args.wrf_in} --wrf_out={args.wrf_out} --tropomi_in={args.tropomi_in}')
    # %% Prep WRF
    wrf_ds = xr.open_dataset(args.wrf_in)
    wrf_ds = fix_time_variable_in_wrf_output(wrf_ds)
    # %% Prep TROPOMI
    tropomi_ds = xr.open_dataset(args.tropomi_in)
    derive_tropomi_no2_pressure_grid(tropomi_ds)
    # %% Minimize the WRF ds size and interpolate in time
    keys = ['PH', 'PHB', 'P', 'PB', 'PSFC', 'ZNW', 'MUB', 'MU'] + ['no2']
    wrf_ds = wrf_ds[keys]
    wrf_ds = wrf_ds.interp(time=tropomi_ds.time, method='linear', kwargs={'bounds_error': True})
    if 'time' not in wrf_ds.dims:  # If time is now a coordinate but not a dimension, put it back. This helps with concatenating later on
        wrf_ds = wrf_ds.expand_dims('time').transpose('time', ...)
    wrf_ds.encoding['unlimited_dims'] = {'time'}
    # %% Deriving intermediate diagnostics
    compute_dz(wrf_ds)
    compute_p(wrf_ds)
    compute_stag_p(wrf_ds)
    calculate_air_mass_dry(wrf_ds)
    # %%
    print('Remember that interpolated NO2 profile will contain NaNs if TROPOMI top is above WRF top')
    wrf_ds['xno2'] = interpolate_wrf_diag_to_tropomi_rho_pressure_grid(wrf_ds, 'no2', tropomi_ds)  # ppmv or 10**6*mol/mol
    da = interpolate_wrf_diag_to_tropomi_rho_pressure_grid(wrf_ds, 'air_mass_dry', tropomi_ds)
    wrf_ds['dvair'] = da / DRY_AIR_MOLAR_MASS  # mol/m2 = kg / m^2 / (kg mol-1)  # dry air column

    trop_layer_index_da = tropomi_ds.tm5_tropopause_layer_index.where(tropomi_ds.qa_value > 0).where((tropomi_ds.tm5_tropopause_layer_index > 0) & (tropomi_ds.tm5_tropopause_layer_index < tropomi_ds.layer.size))
    trop_mask_da = wrf_ds.layer <= trop_layer_index_da
    wrf_ds['trop_no2_column_like_tropomi'] = (10 ** -6 * wrf_ds['xno2'] * wrf_ds['dvair'] * tropomi_ds.averaging_kernel).where(trop_mask_da).sum(dim='layer', min_count=1)  # mol/m2 of no2


    wrf_ds.trop_no2_column_like_tropomi.attrs['long_name'] = 'TROPOMI-like tropospheric NO2, derived from WRF output'
    wrf_ds.trop_no2_column_like_tropomi.attrs['units'] = 'mol/m2'
    # rename to match TROPOMI var exactly
    wrf_ds = wrf_ds.rename_vars({'trop_no2_column_like_tropomi':'nitrogendioxide_tropospheric_column'})
    # %% Save the output
    print('Saving to:\n{}'.format(args.wrf_out))
    os.makedirs(os.path.dirname(args.wrf_out), exist_ok=True)

    export_keys = ['nitrogendioxide_tropospheric_column', ]  # ['trop_no2_column_like_tropomi', ]
    wrf_ds[export_keys].transpose('time', ...).to_netcdf(args.wrf_out)  # , mode=mode, unlimited_dims=unlimited_dim, format='NETCDF4_CLASSIC')
    print('Done')


if __name__ == "__main__":
    # %%
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "--port", "--host", help="pycharm")
    parser.add_argument("--wrf_in", help="wrf input file path")
    parser.add_argument("--wrf_out", help="wrf output file path")
    parser.add_argument("--tropomi_in", help="File path to TROPOMI L2 orbit")
    args = parser.parse_args()
    # %%
    # d01
    # args.wrf_in = '/scratch/osipovs/Data/AirQuality/THOFA/chem_100_v2025.0/wrfout_d01_2023-06-01_00_00_00'
    # args.wrf_out = '/scratch/osipovs/Data/AirQuality/THOFA/chem_100_v2025.0/pp/tropomi_like_no2/wrfout_d01_2023-06-01_00_00_00'
    # args.tropomi_in = '/project/k10048/osipovs/Data/Copernicus/Sentinel-5P/d01/S5P_OFFL_L2__NO2____20230601T081351_20230601T095521_29183_03_020500_20230603T044537.nc'
    #
    # # d02
    # args.wrf_in = '/scratch/osipovs/Data/AirQuality/THOFA/inversion/v5/run_srs_ref/wrfout_d01_2023-06-10_00_00_00'
    # args.wrf_out = '/scratch/osipovs/Data/AirQuality/THOFA/inversion/v5/run_srs_ref/pp/tropomi_like_no2/wrfout_d01_2023-06-10_00_00_00'
    # args.tropomi_in = '/project/k10048/osipovs/Data/Copernicus/Sentinel-5P/THOFA_d02/S5P_OFFL_L2__NO2____20230610T084541_20230610T102711_29311_03_020500_20230612T004757.nc'

    pp_wrf_like_tropomi_no2(args)
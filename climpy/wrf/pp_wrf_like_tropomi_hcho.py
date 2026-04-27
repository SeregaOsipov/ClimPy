from climpy.utils.atmos_utils import DRY_AIR_MOLAR_MASS
# %load_ext autoreload
# %autoreload 2

from scipy import interpolate
import netCDF4
import os
import numpy as np
import xarray as xr
import wrf as wrf
import argparse
from climpy.utils.tropomi_utils import TROPOMI_in_WRF_KEYS, derive_tropomi_hcho_pressure_grid
from climpy.utils.wrf_utils import compute_stag_pressure, compute_stag_z, compute_dz, calculate_air_mass_dry, compute_stag_pressure_impl, compute_p, compute_stag_p, average_wrf_diag_between_tropomi_staggered_pressure_grid, interpolate_wrf_diag_to_tropomi_rho_pressure_grid, generate_netcdf_uniform_time_data, generate_xarray_uniform_time_data, fix_time_variable_in_wrf_output
from wrf import Constants
import datetime as dt

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'


'''
Script derives TROPOMI-specific diagnostics to enable WRF-Chem-TROPOMI comparison.
'''


def pp_wrf_like_tropomi_hcho(args):
    print('pp_wrf_like_tropomi_hcco')
    print(f'--wrf_in={args.wrf_in} --wrf_out={args.wrf_out} --tropomi_in={args.tropomi_in}')
    # %% Prep WRF
    wrf_ds = xr.open_dataset(args.wrf_in)
    wrf_ds = fix_time_variable_in_wrf_output(wrf_ds)
    # %% Prep TROPOMI
    tropomi_ds = xr.open_dataset(args.tropomi_in)
    derive_tropomi_hcho_pressure_grid(tropomi_ds)
    # %% Minimize the WRF ds size and interpolate in time
    keys = ['PH', 'PHB', 'P', 'PB', 'PSFC', 'ZNW', 'MUB', 'MU'] + ['hcho']
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
    print('Remember that interpolated HCHO profile will contain NaNs if TROPOMI top is above WRF top')
    da = interpolate_wrf_diag_to_tropomi_rho_pressure_grid(wrf_ds, 'air_mass_dry', tropomi_ds)
    wrf_ds['dvair'] = da / DRY_AIR_MOLAR_MASS  # mol/m2 = kg / m^2 / (kg mol-1)  # dry air column
    wrf_ds['xhcho'] = interpolate_wrf_diag_to_tropomi_rho_pressure_grid(wrf_ds, 'hcho', tropomi_ds)  # ppmv or 10**6*mol/mol

    # HCHO is tropospheric column
    # We should filter by tropopause layer if available in tropomi_ds (like NO2)
    # If not, we might process all layers or assume HCHO is mostly tropospheric.
    # TM5 products usually have tm5_tropopause_layer_index.
    
    trop_layer_index_da = tropomi_ds.tm5_tropopause_layer_index.where(tropomi_ds.qa_value > 0).where((tropomi_ds.tm5_tropopause_layer_index > 0) & (tropomi_ds.tm5_tropopause_layer_index < tropomi_ds.layer.size))
    trop_mask_da = wrf_ds.layer <= trop_layer_index_da
    wrf_ds['trop_hcho_column_like_tropomi'] = (10 ** -6 * wrf_ds['xhcho'] * wrf_ds['dvair'] * tropomi_ds.averaging_kernel).where(trop_mask_da).sum(dim='layer', min_count=1)  # mol/m2 of hcho
    wrf_ds.trop_hcho_column_like_tropomi.attrs['long_name'] = 'TROPOMI-like tropospheric HCHO, derived from WRF output'
    wrf_ds.trop_hcho_column_like_tropomi.attrs['units'] = 'mol/m2'
    # rename to match TROPOMI var exactly
    wrf_ds = wrf_ds.rename_vars({'trop_hcho_column_like_tropomi':'formaldehyde_tropospheric_vertical_column'})
    # %% Save the output
    print('Saving to:\n{}'.format(args.wrf_out))
    os.makedirs(os.path.dirname(args.wrf_out), exist_ok=True)

    export_keys = ['formaldehyde_tropospheric_vertical_column', ]
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

    pp_wrf_like_tropomi_hcho(args)

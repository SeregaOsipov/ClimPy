import argparse
import os

import xarray as xr

from climpy.utils.atmos_utils import DRY_AIR_MOLAR_MASS
from climpy.utils.tropomi_utils import derive_tropomi_chocho_pressure_grid
from climpy.utils.wrf_utils import compute_dz, calculate_air_mass_dry, compute_p, compute_stag_p, interpolate_wrf_diag_to_tropomi_rho_pressure_grid, fix_time_variable_in_wrf_output

# %load_ext autoreload
# %autoreload 2

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'


'''
Script derives TROPOMI-specific diagnostics to enable WRF-Chem-TROPOMI comparison.
'''


def pp_wrf_like_tropomi_chocho(args):
    print('pp_wrf_like_tropomi_chocho')
    print(f'--wrf_in={args.wrf_in} --wrf_out={args.wrf_out} --tropomi_in={args.tropomi_in}')
    # %% Prep WRF
    wrf_ds = xr.open_dataset(args.wrf_in)
    wrf_ds = fix_time_variable_in_wrf_output(wrf_ds)
    # %% Prep TROPOMI
    tropomi_ds = xr.open_dataset(args.tropomi_in)
    derive_tropomi_chocho_pressure_grid(tropomi_ds)
    # %% Minimize the WRF ds size and interpolate in time
    keys = ['PH', 'PHB', 'P', 'PB', 'PSFC', 'ZNW', 'MUB', 'MU'] + ['gly']
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
    print('Remember that interpolated chocho profile will contain NaNs if TROPOMI top is above WRF top')
    da = interpolate_wrf_diag_to_tropomi_rho_pressure_grid(wrf_ds, 'air_mass_dry', tropomi_ds)
    wrf_ds['dvair'] = da / DRY_AIR_MOLAR_MASS  # mol/m2 = kg / m^2 / (kg mol-1)  # dry air column
    wrf_ds['xchocho'] = interpolate_wrf_diag_to_tropomi_rho_pressure_grid(wrf_ds, 'gly', tropomi_ds)  # ppmv or 10**6*mol/mol

    wrf_ds['trop_chocho_column_like_tropomi'] = (10 ** -6 * wrf_ds['xchocho'] * wrf_ds['dvair'] * tropomi_ds.averaging_kernel).sum(dim='layer', min_count=1)  # mol/m2 of chocho
    wrf_ds.trop_chocho_column_like_tropomi.attrs['long_name'] = 'TROPOMI-like tropospheric chocho, derived from WRF output'
    wrf_ds.trop_chocho_column_like_tropomi.attrs['units'] = 'mol/m2'
    # rename to match TROPOMI var exactly
    wrf_ds = wrf_ds.rename_vars({'trop_chocho_column_like_tropomi':'glyoxal_tropospheric_vertical_column'})
    # %% Save the output
    print('Saving to:\n{}'.format(args.wrf_out))
    os.makedirs(os.path.dirname(args.wrf_out), exist_ok=True)

    export_keys = ['glyoxal_tropospheric_vertical_column', ]
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

    pp_wrf_like_tropomi_chocho(args)

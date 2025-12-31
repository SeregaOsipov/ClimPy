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
from climpy.utils.tropomi_utils import TROPOMI_in_WRF_KEYS, derive_tropomi_so2_pressure_grid
from climpy.utils.wrf_utils import compute_stag_pressure, compute_stag_z, compute_dz, calculate_air_mass_dry, compute_stag_pressure_impl, compute_p, compute_stag_p, average_wrf_diag_between_tropomi_staggered_pressure_grid, interpolate_wrf_diag_to_tropomi_rho_pressure_grid, generate_netcdf_uniform_time_data, generate_xarray_uniform_time_data
from wrf import Constants
import datetime as dt

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'


'''
Script derives TROPOMI-specific diagnostics to enable WRF-Chem-TROPOMI comparison.
'''


def pp_wrf_like_tropomi_so2(args):
    print('Will process this WRF:\nin {}\nout {}'.format(args.wrf_in, args.wrf_out))
    # %% Prep WRF
    wrf_ds = xr.open_dataset(args.wrf_in)
    if 'XTIME' in wrf_ds.dims:
        wrf_ds = wrf_ds.rename({'XTIME': 'Time'}).rename({'Time': 'time'})
    else:
        wrf_ds['time'] = generate_xarray_uniform_time_data(wrf_ds.Times)
        wrf_ds = wrf_ds.rename({'Time': 'time'})
    # %% Prep TROPOMI
    tropomi_ds = xr.open_dataset(args.tropomi_in)
    derive_tropomi_so2_pressure_grid(tropomi_ds)
    # %% Minimize the WRF ds size and interpolate in time
    keys = ['PH', 'PHB', 'P', 'PB', 'PSFC', 'ZNW', 'MUB', 'MU'] + ['so2']
    wrf_ds = wrf_ds[keys]
    wrf_ds = wrf_ds.interp(time=tropomi_ds.time, method='linear')
    # %% Deriving intermediate diagnostics
    compute_dz(wrf_ds)
    compute_p(wrf_ds)
    compute_stag_p(wrf_ds)
    calculate_air_mass_dry(wrf_ds)

    derive_tropomi_so2_pressure_grid(tropomi_ds)
    # %%
    print('Remember that interpolated SO2 profile will contain NaNs if TROPOMI top is above WRF top')
    da = interpolate_wrf_diag_to_tropomi_rho_pressure_grid(wrf_ds, 'air_mass_dry', tropomi_ds)
    wrf_ds['dvair'] = da / DRY_AIR_MOLAR_MASS  # mol/m2 = kg / m^2 / (kg mol-1)  # dry air column
    wrf_ds['xso2'] = interpolate_wrf_diag_to_tropomi_rho_pressure_grid(wrf_ds, 'so2', tropomi_ds)  # ppmv or 10**6*mol/mol

    # SO2 uses total column, so we might sum over all layers.
    # However, we should be careful if TROPOMI provides a tropopause layer index and we want to be consistent with NO2.
    # But for SO2 total column, usually all layers are considered.
    # If a mask is needed, it should be based on valid data or averaging kernel scope.
    # We will assume averaging kernel covers the relevant atmosphere.

    wrf_ds['trop_so2_column_like_tropomi'] = (10 ** -6 * wrf_ds['xso2'] * wrf_ds['dvair'] * tropomi_ds.averaging_kernel).sum(dim='layer')  # mol/m2 of so2
    wrf_ds.trop_so2_column_like_tropomi.attrs['long_name'] = 'TROPOMI-like SO2, derived from WRF output'
    wrf_ds.trop_so2_column_like_tropomi.attrs['units'] = 'mol/m2'
    # rename to match TROPOMI var exactly
    wrf_ds = wrf_ds.rename_vars({'trop_so2_column_like_tropomi':'sulfurdioxide_total_vertical_column'})
    # %% Save the output
    print('Saving to:\n{}'.format(args.wrf_out))
    os.makedirs(os.path.dirname(args.wrf_out), exist_ok=True)

    export_keys = ['sulfurdioxide_total_vertical_column', ]
    wrf_ds[export_keys].to_netcdf(args.wrf_out)  # , mode=mode, unlimited_dims=unlimited_dim, format='NETCDF4_CLASSIC')
    print('Done')


if __name__ == "__main__":
    # %%
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "--port", "--host", help="pycharm")
    parser.add_argument("--wrf_in", help="wrf input file path")
    parser.add_argument("--wrf_out", help="wrf output file path")
    parser.add_argument("--tropomi_in", help="File path to TROPOMI L2 orbit")
    args = parser.parse_args()

    pp_wrf_like_tropomi_so2(args)

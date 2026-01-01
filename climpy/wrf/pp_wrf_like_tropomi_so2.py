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

TROPOMI User Guide: https://sentiwiki.copernicus.eu/__attachments/1673595/S5P-L2-DLR-PUM-400E%20-%20Sentinel-5P%20Level%202%20Product%20User%20Manual%20Sulphur%20Dioxide%20SO2%202024%20-%202.8.0.pdf#page=34.09
'''


def pp_wrf_like_tropomi_so2(args):
    print('Will process this WRF:\nin {}\nout {}'.format(args.wrf_in, args.wrf_out))
    # %% Prep WRF
    wrf_ds = xr.open_dataset(args.wrf_in)
    if 'XTIME' in wrf_ds.dims:
        wrf_ds = wrf_ds.rename({'XTIME': 'time'})
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
    # %% Derive TROPOMI-like diagnostic
    # this SO2 product does not provide enough info to build staggered grid
    # wrf_ds['xso2'] = average_wrf_diag_between_tropomi_staggered_pressure_grid(wrf_ds, 'so2', tropomi_ds)
    # wrf_ds['dvair'] = average_wrf_diag_between_tropomi_staggered_pressure_grid(wrf_ds, 'air_mass_dry', tropomi_ds)
    # So we interpolate to rho-grid
    wrf_ds['xso2'] = interpolate_wrf_diag_to_tropomi_rho_pressure_grid(wrf_ds, 'so2', tropomi_ds)  # ppmv or 10**6*mol/mol
    wrf_ds['dvair'] = interpolate_wrf_diag_to_tropomi_rho_pressure_grid(wrf_ds, 'air_mass_dry', tropomi_ds)
    wrf_ds['dvair'] /= DRY_AIR_MOLAR_MASS  # mol/m2 = kg / m^2 / (kg mol-1)  # dry air column

    wrf_ds['dvso2'] = 10 ** -6 * wrf_ds['xso2'] * wrf_ds['dvair']  # mol/m2 of so2
    wrf_ds['vso2'] = tropomi_ds.sulfurdioxide_profile_apriori.sum(dim='layer') + (tropomi_ds.averaging_kernel * (wrf_ds['dvso2'] - tropomi_ds.sulfurdioxide_profile_apriori)).sum(dim='layer')
    # wrf_ds['vso2'] = (10 ** -6 * wrf_ds['xso2'] * wrf_ds['dvair'] * tropomi_ds.averaging_kernel).sum(dim='layer')  # mol/m2 of so2  # no a priori profile
    wrf_ds['xso2_like_tropomi'] = wrf_ds.vso2

    # wrf_ds['xso2_like_tropomi'] = 10 ** 9 * wrf_ds['vso2'] / wrf_ds['dvair'].sum('layer')  # ppbv. This is converts back to volume mixing ratio
    # wrf_ds.xso2_like_tropomi.attrs['units'] = '1e-9'  # ppbv

    wrf_ds.xso2_like_tropomi.attrs['long_name'] = 'TROPOMI-like SO2 total column, derived from WRF output'
    wrf_ds.xso2_like_tropomi.attrs['units'] = 'mol/m2'
    # rename to match TROPOMI var exactly
    wrf_ds = wrf_ds.rename_vars({'xso2_like_tropomi': 'sulfurdioxide_total_vertical_column'})
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

    #%% Debug
    # d02
    # args.wrf_in = '/scratch/osipovs/Data/AirQuality/THOFA/inversion/v5/run_srs_ref/wrfout_d01_2023-06-01_00_00_00'
    # args.wrf_out = '/scratch/osipovs/Data/AirQuality/THOFA/inversion/v5/run_srs_ref/pp/tropomi_like_so2/wrfout_d01_2023-06-01_00_00_00'
    # args.tropomi_in = '/project/k10048/osipovs//Data/Copernicus/Sentinel-5P//THOFA_d02/S5P_OFFL_L2__SO2____20230601T081351_20230601T095521_29183_03_020401_20230603T102223.nc'

    #%%
    pp_wrf_like_tropomi_so2(args)

# %load_ext autoreload
# %autoreload 2

import argparse
import os

import xarray as xr

from climpy.utils.atmos_utils import DRY_AIR_MOLAR_MASS
from climpy.utils.tropomi_utils import derive_tropomi_co_pressure_grid
from climpy.utils.wrf_utils import compute_dz, calculate_air_mass_dry, compute_p, compute_stag_p, generate_xarray_uniform_time_data, interpolate_wrf_diag_to_tropomi_rho_pressure_grid, fix_time_variable_in_wrf_output

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

'''
Script derives TROPOMI-specific diagnostics to enable WRF-Chem-TROPOMI comparison.

# Individual run
wrf_in=/scratch/osipovs/Data/AirQuality/THOFA/inversion/v5/run_srs_ref/wrfout_d01_2023-06-10_00_00_00
wrf_out=/scratch/osipovs/Data/AirQuality/THOFA/inversion/v5/run_srs_ref/pp/tropomi_like_co/wrfout_d01_2023-06-10_00_00_00
tropomi_in=/project/k10048/osipovs/Data/Copernicus/Sentinel-5P/THOFA_d02/S5P_OFFL_L2__CO____.nc
python -u ${CLIMPY}climpy/wrf/pp_wrf_like_tropomi_co.py --wrf_in=${wrf_in} --wrf_out=${wrf_out} --tropomi_in=${tropomi_in}
'''


def pp_wrf_like_tropomi_co(args):
    print('pp_wrf_like_tropomi_co')
    print(f'--wrf_in={args.wrf_in} --wrf_out={args.wrf_out} --tropomi_in={args.tropomi_in}')
    # %% Prep WRF
    wrf_ds = xr.open_dataset(args.wrf_in)
    wrf_ds = fix_time_variable_in_wrf_output(wrf_ds)
    # %% Prep TROPOMI
    tropomi_ds = xr.open_dataset(args.tropomi_in)
    derive_tropomi_co_pressure_grid(tropomi_ds)
    # %% Minimize the WRF ds size and interpolate in time
    keys = ['PH', 'PHB', 'P', 'PB', 'PSFC', 'ZNW', 'MUB', 'MU'] + ['co']
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
    wrf_ds['xco'] = interpolate_wrf_diag_to_tropomi_rho_pressure_grid(wrf_ds, 'co', tropomi_ds)
    wrf_ds['dvair'] = interpolate_wrf_diag_to_tropomi_rho_pressure_grid(wrf_ds, 'air_mass_dry', tropomi_ds)
    wrf_ds['dvair'] /= DRY_AIR_MOLAR_MASS  # mol/m2 = kg / m^2 / (kg mol-1)  # dry air column

    wrf_ds['dvco'] = 10 ** -6 * wrf_ds['xco'] * wrf_ds['dvair']  # mol/m2 of carbon monoxide
    wrf_ds['vco'] = tropomi_ds.carbonmonoxide_profile_apriori.sum(dim='layer', min_count=1) + (tropomi_ds.column_averaging_kernel * (wrf_ds['dvco'] - tropomi_ds.carbonmonoxide_profile_apriori)).sum(dim='layer', min_count=1)  # use min_count to preserve missing values

    wrf_ds['xco_like_tropomi'] = wrf_ds['vco'] # mol/m2
    wrf_ds.xco_like_tropomi.attrs['units'] = 'mol/m2'
    # rename to match TROPOMI var exactly
    wrf_ds = wrf_ds.rename_vars({'xco_like_tropomi': 'carbonmonoxide_total_column_corrected'})
    # %% Save the output
    print('Saving to:\n{}'.format(args.wrf_out))
    os.makedirs(os.path.dirname(args.wrf_out), exist_ok=True)

    export_keys = ['carbonmonoxide_total_column_corrected', ]
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

    pp_wrf_like_tropomi_co(args)

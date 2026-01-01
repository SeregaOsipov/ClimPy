from climpy.utils.atmos_utils import DRY_AIR_MOLAR_MASS
import os
import numpy as np
import xarray as xr
import argparse
from climpy.utils.tropomi_utils import derive_tropomi_o3_pr_pressure_grid
from climpy.utils.wrf_utils import compute_dz, compute_p, compute_stag_p, calculate_air_mass_dry, interpolate_wrf_diag_to_tropomi_rho_pressure_grid, generate_xarray_uniform_time_data

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'


'''
Script derives TROPOMI-specific diagnostics to enable WRF-Chem-TROPOMI comparison for Ozone Profile.
'''


def pp_wrf_like_tropomi_o3_profile(args):
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
    derive_tropomi_o3_pr_pressure_grid(tropomi_ds)

    # %% Minimize the WRF ds size and interpolate in time
    keys = ['PH', 'PHB', 'P', 'PB', 'PSFC', 'ZNW', 'MUB', 'MU'] + ['o3']
    wrf_ds = wrf_ds[keys]
    wrf_ds = wrf_ds.interp(time=tropomi_ds.time, method='linear')

    # %% Deriving intermediate diagnostics
    compute_dz(wrf_ds)
    compute_p(wrf_ds)
    compute_stag_p(wrf_ds)
    calculate_air_mass_dry(wrf_ds)

    # Re-derive tropomi pressure grid to ensure consistency
    derive_tropomi_o3_pr_pressure_grid(tropomi_ds)

    # %% Interpolate WRF to TROPOMI vertical grid
    print('Remember that interpolated O3 profile will contain NaNs if TROPOMI top is above WRF top')

    # 1. Interpolate dry air mass to TROPOMI grid to get layer air mass
    da = interpolate_wrf_diag_to_tropomi_rho_pressure_grid(wrf_ds, 'air_mass_dry', tropomi_ds)
    wrf_ds['dvair'] = da / DRY_AIR_MOLAR_MASS  # mol/m2

    # 2. Interpolate O3 mixing ratio to TROPOMI grid
    # WRF o3 is ppmv
    wrf_ds['xo3'] = interpolate_wrf_diag_to_tropomi_rho_pressure_grid(wrf_ds, 'o3', tropomi_ds)  # ppmv

    # 3. Calculate WRF O3 partial column (mol m-2)
    # partial column = mixing ratio (mol/mol) * total air column (mol/m2)
    wrf_ds['dvo3'] = 10**-6 * wrf_ds['xo3'] * wrf_ds['dvair'] # mol/m2

    # 4. Apply Averaging Kernel
    # x_hat = x_a + A * (x_true - x_a)

    if 'averaging_kernel' in tropomi_ds and 'ozone_profile_apriori' in tropomi_ds:
        ak = tropomi_ds.averaging_kernel
        apriori = tropomi_ds.ozone_profile_apriori
        diff = wrf_ds['dvo3'] - apriori

        # Apply Averaging Kernel using numpy einsum for robustness
        # Assuming AK dims are (..., layer, layer). The last dimension sums over the profile.
        # We process values to avoid dimension name conflicts in xarray if they are identical.

        # Broadcasting logic:
        # AK: (time, lat, lon, layer_out, layer_in) or similar.
        # diff: (time, lat, lon, layer_in)
        # Result: (time, lat, lon, layer_out)

        # We assume the last two dimensions of AK are the matrix (out, in) or (row, col).
        # Standard definition: A[i,j] = dx_i / dx_true_j.
        # So we sum over the last index of AK and the last index of diff.

        ak_values = ak.values
        diff_values = diff.values

        # Handle cases where diff might have different shape prefix (e.g. broadcasting time)
        # But here wrf_ds was interpolated to tropomi_ds time, so shapes should align.

        # Numpy einsum: '...ij,...j->...i'
        try:
            term_values = np.einsum('...ij,...j->...i', ak_values, diff_values)

            # Create DataArray for the term
            # Use dimensions from apriori (which matches output profile)
            term = xr.DataArray(term_values, coords=apriori.coords, dims=apriori.dims)

            wrf_ds['ozone_profile_like_tropomi'] = apriori + term
        except Exception as e:
            print(f"Error applying averaging kernel: {e}. Returning raw profile.")
            wrf_ds['ozone_profile_like_tropomi'] = wrf_ds['dvo3']
    else:
        print("Warning: averaging_kernel or ozone_profile_apriori not found. Returning raw interpolated profile.")
        wrf_ds['ozone_profile_like_tropomi'] = wrf_ds['dvo3']

    wrf_ds.ozone_profile_like_tropomi.attrs['long_name'] = 'TROPOMI-like Ozone Profile, derived from WRF output'
    wrf_ds.ozone_profile_like_tropomi.attrs['units'] = 'mol/m2'

    # rename to match TROPOMI var exactly
    wrf_ds = wrf_ds.rename_vars({'ozone_profile_like_tropomi':'ozone_profile'})

    # %% Save the output
    print('Saving to:\n{}'.format(args.wrf_out))
    os.makedirs(os.path.dirname(args.wrf_out), exist_ok=True)

    export_keys = ['ozone_profile', ]
    wrf_ds[export_keys].to_netcdf(args.wrf_out)
    print('Done')


if __name__ == "__main__":
    # %%
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "--port", "--host", help="pycharm")
    parser.add_argument("--wrf_in", help="wrf input file path")
    parser.add_argument("--wrf_out", help="wrf output file path")
    parser.add_argument("--tropomi_in", help="File path to TROPOMI L2 orbit")
    args = parser.parse_args()

    pp_wrf_like_tropomi_o3_profile(args)

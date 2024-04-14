import numpy as np
import xarray as xr


def derive_merra2_pressure_profile(ds, hPa=False):
    '''
    See this doc for details on Vertical Structure https://gmao.gsfc.nasa.gov/pubs/docs/Bosilovich785.pdf
    :param ds:
    :return:
    '''

    # Build the entire 3d field first
    layer_pressure_thickness = ds['DELP'].sortby('lev')  # sort to make sure that summation takes places from top

    # bug preventive measure. Check that pressure sorting will not reverse relative to the parent array (MERRA sort from 1 layer at TOA to last layer at BOA)
    if not all(ds['DELP'].lev.to_numpy() == layer_pressure_thickness.lev.to_numpy()):  # in MERRA2 first layer by index is TOA
        raise Exception('merra_utils:derive_merra2_pressure_profile. Pressure profile will likely be reversed')
        # in this case precompute pressure first and them flip the entire MERRA2 df

    # summation has to start from the top, p_top is fixed to 1 Pa
    pressure_stag_no_toa = 1 + layer_pressure_thickness.cumsum(dim='lev')

    # xarray implementation
    toa_pressure = pressure_stag_no_toa.sel(lev=1, drop=False).copy(deep=True)  # MERRA2 has first level at TOA
    toa_pressure[:] = 1
    toa_pressure['lev'] = 0
    pressure_stag = xr.concat([toa_pressure, pressure_stag_no_toa], dim='lev')
    # pressure_stag = pressure_stag.transpose('lev', ...)  # xr concat changes dimensions order

    # derive the pressure at the rho grid from stag grid
    pressure_rho = pressure_stag.rolling(lev=2).mean().dropna('lev')

    pressure_rho = pressure_rho.rename('p_rho')
    pressure_stag = pressure_stag.rename('p_stag')

    pressure_stag.attrs['long_name'] = 'Pressure'
    pressure_rho.attrs['long_name'] = 'Pressure'

    if hPa:
        pressure_stag /= 10**2
        pressure_stag.attrs['units'] = 'hPa'

        pressure_rho /= 10**2
        pressure_rho.attrs['units'] = 'hPa'

    return pressure_stag, pressure_rho



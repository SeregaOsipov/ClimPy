import numpy as np
import xarray as xr


def derive_merra2_pressure_profile(merra_ds, hPa=False):
    '''
    See this doc for details on Vertical Structure https://gmao.gsfc.nasa.gov/pubs/docs/Bosilovich785.pdf

    Rename lev to level

    :param merra_ds:
    :return:
    '''

    # Build the entire 3d field first
    layer_pressure_thickness = merra_ds['DELP'].sortby('level')  # sort to make sure that summation takes places from top

    # bug preventive measure. Check that pressure sorting will not reverse relative to the parent array (MERRA sort from 1 layer at TOA to last layer at BOA)
    if not all(merra_ds['DELP'].level.to_numpy() == layer_pressure_thickness.level.to_numpy()):  # in MERRA2 first layer by index is TOA
        raise Exception('merra_utils:derive_merra2_pressure_profile. Pressure profile will likely be reversed')
        # in this case precompute pressure first and them flip the entire MERRA2 df

    # summation has to start from the top, p_top is fixed to 1 Pa
    pressure_stag_no_toa = 1 + layer_pressure_thickness.cumsum(dim='level')

    # xarray implementation
    toa_pressure = pressure_stag_no_toa.sel(level=1, drop=False).copy(deep=True)  # MERRA2 has first level at TOA
    toa_pressure[:] = 1
    toa_pressure['level'] = 0
    pressure_stag = xr.concat([toa_pressure, pressure_stag_no_toa], dim='level')
    # pressure_stag = pressure_stag.transpose('level', ...)  # xr concat changes dimensions order

    # derive the pressure at the rho grid from stag grid
    pressure_rho = pressure_stag.rolling(level=2).mean().dropna('level')

    pressure_rho = pressure_rho.rename('p_rho')
    pressure_stag = pressure_stag.rename('p_stag')

    pressure_stag.attrs['long_name'] = 'Pressure'
    pressure_rho.attrs['long_name'] = 'Pressure'

    if hPa:
        pressure_stag /= 10**2
        pressure_stag.attrs['units'] = 'hPa'

        pressure_rho /= 10**2
        pressure_rho.attrs['units'] = 'hPa'

    merra_ds['p_stag'] = pressure_stag
    merra_ds['p_rho'] = pressure_rho

    return pressure_stag, pressure_rho


def generate_merra2_file_name(merra2_file_name_template, date, stream):
    year = date.year
    month = date.month
    if year == 2021 and 6 <= month <= 9:
        vvv = 401
    elif year >= 2011:
        vvv = 400
    elif year >= 2001:
        vvv = 300
    elif year >= 1992:
        vvv = 200
    else:
        vvv = 100

    file_name = merra2_file_name_template.format(date_time=date.strftime('%Y%m%d'), stream=stream, VVV=vvv)
    return file_name
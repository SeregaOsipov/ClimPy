import functools
import netCDF4
import numpy as np
from climpy.utils.file_path_utils import get_root_storage_path_on_hpc

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'


def interpolate_ri_in_wavelength(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        wavelengths = None
        if 'wavelengths' in kwargs:
            wavelengths = kwargs.pop('wavelengths')

        vo = func(*args, **kwargs)

        if wavelengths is not None:
            # interpolate RI onto internal wavelength grid
            ri = np.interp(wavelengths, vo['wl'], vo['ri'])  # ri = real+1j*imag
            vo['ri'] = ri
            vo['wl'] = wavelengths

        return vo
    return wrapper_decorator


@interpolate_ri_in_wavelength
def get_Williams_Palmer_refractive_index():
    '''
    :return: Refractive index from Palmer and Williams(1975) of 75 % sulfate aerosol
    wavelength is in microns
    '''

    nc_fp = get_root_storage_path_on_hpc() + '/Data/Harvard/HITRAN/HITRAN2012/Aerosols/netcdf/palmer_williams_h2so4.nc'
    nc = netCDF4.Dataset(nc_fp)
    wl = nc.variables['wavelength'][:]
    ri_real = nc.variables['rn'][3, :]  # index 4 is 75% solution
    ri_imag = nc.variables['ri'][3, :]

    ri_vo = {}
    ri_vo['ri'] = ri_real + 1j * ri_imag
    ri_vo['ri'] = ri_vo['ri'][::-1]
    ri_vo['wl'] = wl[::-1]
    return ri_vo


@interpolate_ri_in_wavelength
def get_dust_ri():
    '''
    SW: https://data.eurochamp.org/data-access/optical-properties/
    LW: https://acp.copernicus.org/articles/17/1901/2017/
    :return:
    '''

    wl = np.array([0.52, 0.95, 10])
    ri_real = np.array([1.54, 1.54, 1.602])
    ri_imag = np.array([15*10**-4, 6*10**-4, 0.28])

    ri_vo = {}
    ri_vo['ri'] = ri_real + 1j * ri_imag
    ri_vo['wl'] = wl
    return ri_vo
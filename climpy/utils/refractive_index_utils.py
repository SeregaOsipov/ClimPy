__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

import netCDF4
import numpy as np


def get_Williams_Palmer_refractive_index():
    '''
    :return: Refractive index from Palmer and Williams(1975) of 75 % sulfate aerosol
    wavelength is in microns
    '''

    nc_fp = '/home/osipovs/Data/Harvard/HITRAN/HITRAN2012/Aerosols/netcdf/palmer_williams_h2so4.nc'
    nc = netCDF4.Dataset(nc_fp)
    wl = nc.variables['wavelength'][:]
    ri_real = nc.variables['rn'][3, :]  # index 4 is 75% solution
    ri_imag = nc.variables['ri'][3, :]

    ri_vo = {}
    ri_vo['ri'] = ri_real + 1j * ri_imag
    ri_vo['wl'] = wl
    return ri_vo


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
__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

import netCDF4


def get_Williams_Palmer_refractive_index():
    # Refractive index from Palmer and Williams(1975) of 75 % sulfate aerosol
    # wavelength in microns
    nc_fp = '/home/osipovs/Data/Harvard/HITRAN/HITRAN2012/Aerosols/netcdf/palmer_williams_h2so4.nc'
    nc = netCDF4.Dataset(nc_fp)
    wl = nc.variables['wavelength'][:]
    ri_real = nc.variables['rn'][3, :]  # index 4 is 75% solution
    ri_imag = nc.variables['ri'][3, :]

    ri_vo = {}
    ri_vo['ri'] = ri_real + 1j * ri_imag
    ri_vo['wl'] = wl
    return ri_vo


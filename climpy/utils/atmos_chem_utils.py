import os

import numpy as np

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

import xarray as xr

from climpy.utils.lblrtm_utils import Gas


def get_ozone_quantum_yield(wavelengths, temperature):
    """

    :param wavelengths: in microns
    :param temperature: in Kelvin
    :return:
    """
    wavelengths_nm = wavelengths * 10 ** 3

    qy = np.empty(wavelengths.shape)
    qy[:] = np.NaN

    # In the wavelength range 329 – 340 nm the recommended value of Φ(O1D) = 0.08 ± 0.04, independent of temperature.
    ind = np.logical_and(wavelengths > 0.328, wavelengths <= 0.340)
    qy[ind] = 0.08
    # For λ > 340 nm, the quantum yield may be non-zero but no recommendation is made
    ind = wavelengths > 0.340
    qy[ind] = 0
    print('O3 quantum yield data might be non zero λ > 340 nm, no recommendations were done for this region')
    # For the wavelength range 220 – 305 nm, the recommended O(1D) quantum yield is 0.90, independent of temperature,
    # based on the study by Takahashi et al. [141] and the review by Matsumi and Kawasaki [100].
    ind = np.logical_and(wavelengths >= 0.220, wavelengths < 0.306)
    qy[ind] = 0.9
    # A simple expression in the wavelength range 193 – 225 nm was derived ΦO1D (λ) = 1.37 x 10-2 λ - 2.16
    ind = np.logical_and(wavelengths >= 0.193, wavelengths <= 0.225)
    qy[ind] = 1.37 * 10**-2 * wavelengths_nm[ind] - 2.16

    # following the Chemical Kinetics and Photochemical Data for Use in Atmospheric Studies Evaluation Number 17
    # available at http://jpldataeval.jpl.nasa.gov/pdf/JPL%2010-6%20Final%2015June2011.pdf
    # page 4a-11
    ind = np.logical_and(wavelengths >= 0.306, wavelengths <= 0.328)

    # table data
    X1 = 304.225; X2 = 314.957; X3 = 310.737
    omega1 = 5.576; omega2 = 6.601; omega3 = 2.187
    A1 = 0.8036; A2 = 8.9061; A3 = 0.1192
    nu1 = 0; nu2 = 825.518
    c = 0.0765
    R = 0.695

    q1 = np.exp(-nu1 / R/ temperature)
    q2 = np.exp(-nu2 / R/ temperature)

    qy_item = q1/(q1+q2) * A1 * np.exp( -((X1-wavelengths_nm[ind]) / omega1)**4 ) +\
              q2/(q1+q2) * A2 * (temperature / 300)**2 * np.exp( -((X2-wavelengths_nm[ind]) / omega2)** 2 ) +\
              A3 * (temperature / 300)** 1.5 * np.exp( -((X3-wavelengths_nm[ind]) / omega3)**2 ) + c


    qy[ind] = qy_item

    qy_vo = {}
    qy_vo['wavelengths'] = wavelengths
    qy_vo['quantum_yield'] = qy

    return qy_vo


def get_atmospheric_gases_composition():
    '''
    Prepare the atmospheric chem composition for LBLRTM and DISORT

    LBLRTM gas units:

    JCHAR = 1-6           - default to value for specified model atmosphere
      = " ",A         - volume mixing ratio (ppmv):
      = B             - number density (cm-3)
      = C             - mass mixing ratio (gm/kg)
      = D             - mass density (gm m-3)
      = E             - partial pressure (mb)
      = F             - dew point temp (K) *H2O only*
      = G             - dew point temp (C) *H2O only*
      = H             - relative humidity (percent) *H2O only*
      = I             - available for user definition

    '''
    fp = '/work/mm0062/b302074/Data/NASA/GMI/gmiClimatology.nc'  # GMI climatology
    fp = os.path.expanduser('~') + '/Data/NASA/GMI/gmiClimatology.nc'  # GMI climatology
    ds = xr.open_dataset(fp)
    # fix the metadata
    ds = ds.set_coords({'lat', 'lon', 'level'})
    ds = ds.swap_dims({'latitude_dim': 'lat', 'longitude_dim': 'lon', 'eta_dim': 'level'})
    ds = ds.rename({'species_dim': 'species'})

    species = []  # derive labels from netcdf
    for index in range(ds.const_labels.shape[1]):
        label = ''.join([r.item().decode().strip() for r in ds.const_labels[:,index]])
        species += [label, ]

    ds['species'] = (('species', ), species)

    # SO2 is missing, setup dummy profile.
    so2_ds = ds.sel(species=Gas.O2.value)  # Use O2 to make a single gas copy as a template for SO2
    so2_ds.const[:]=0
    so2_ds['species'] = (('species',), ['SO2',])

    # CO2 is missing, setup 388.5 (as of june 2012).
    co2_ds = ds.sel(species=Gas.O2.value)  # Use O2 to make a single gas copy as a template for SO2
    co2_ds.const[:] = 388.5  # units A
    co2_ds['species'] = (('species',), ['CO2', ])

    ds = xr.concat([ds, so2_ds, co2_ds], 'species')
    ds['time'] = np.arange(12)  # zero based indexing

    ds = ds.drop(labels=('const_labels',))

    # add units variable according to LBLRTM
    ds['units'] = (('species',), ['A',]*ds.species.size)
    ds.units.attrs['long_name'] = 'units according to LBLRTM'

    return ds



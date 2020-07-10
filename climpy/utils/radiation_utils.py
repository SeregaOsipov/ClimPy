__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

import numpy as np


def get_photon_energy(wavelengths):
    """
    computes the energy of the photon of a given wavelength
    :param wavelengths: [um]
    :return: units are W *m^-2 * um^-1  = J * s^-1 *m^-2 *um^-1
    """
    plank_constant = 6.62606957 * 10**-34  # J*s
    speed_of_light = 299792458  # m*s^-1
    nu = speed_of_light / wavelengths  # s^-1
    E = plank_constant * nu  # J
    return E


def integrate_spectral_flux(wavelengths, spectral_flux, wl_min=0, wl_max=np.inf):
    """
    Integrate spectral flux in wavelength
    :param wavelengths: um
    :param spectral_flux: as returned from disort (my impl) [W * m^-2 * um^-1]
    :return: W m^-2
    """

    ind = np.logical_and(wavelengths >= wl_min, wavelengths <= wl_max)
    broadband_flux = np.trapz(spectral_flux[ind], wavelengths[ind], axis=0)

    return broadband_flux

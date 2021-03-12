import numpy as np
from climpy.utils.jpl_utils import parse_absorption_cross_section_file

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

from climpy.utils.quantum_yeild_utils import get_ozone_quantum_yield

from climpy.utils.radiation_utils import get_photon_energy, integrate_spectral_flux

"""
This module contains the routines, which I often use for ozone chemistry analysis
"""


def j_o2_pp(matlab_output_vo):
    o2_xs_vo = parse_absorption_cross_section_file('O2_Yoshino(1988)_298K_205-240nm(rec).txt')
    # xs_vo_o2_fally = parse_absorption_cross_section_file('O2_Fally(2000)_287-289K_240-294nm.txt')

    # The overall quantum yield for photodissociation channel (1), O( 3 P) + O( 3 P), is unity, Φ(1) = 1, for 175 < λ < 242 nm
    o2_qy_vo = {}
    o2_qy_vo['wavelengths'] = o2_xs_vo['wavelengths']
    o2_qy_vo['quantum_yield'] = np.ones(o2_xs_vo['wavelengths'].shape)

    xs_vo = o2_xs_vo
    qy_vo = o2_qy_vo

    # custom range for O2
    wavelengths = matlab_output_vo['wl_grid']
    ind = np.logical_and(wavelengths >= np.min(xs_vo['wavelengths']), wavelengths <= np.max(xs_vo['wavelengths']))
    wavelengths = wavelengths[ind]
    spectral_actinic_flux = matlab_output_vo['spectral_actinic_flux'][ind]

    spectral_j, matlab_output_vo['j_o2'] = compute_j(wavelengths, spectral_actinic_flux, xs_vo, qy_vo)
    # daylight mean
    matlab_output_vo['j_o2_mean'] = np.mean(matlab_output_vo['j_o2'], axis=2)


def j_o3_pp(matlab_output_vo):
    o3_xs_vo = parse_absorption_cross_section_file('O3_JPL-2010(2011)_293-298K_121.6-827.5nm(rec).txt')

    wavelengths = matlab_output_vo['wl_grid']
    ind = np.ones(wavelengths.shape, dtype=bool)
    # ind = np.logical_and(wavelengths >= np.min(xs_vo['wavelengths']), wavelengths <= np.max(xs_vo['wavelengths']))
    wavelengths = wavelengths[ind]

    qy_vo_o3 = get_ozone_quantum_yield(wavelengths, 253)

    xs_vo = o3_xs_vo
    qy_vo = qy_vo_o3

    spectral_actinic_flux = matlab_output_vo['spectral_actinic_flux'][ind]
    spectral_j, matlab_output_vo['j_o3'] = compute_j(wavelengths, spectral_actinic_flux, xs_vo, qy_vo)
    # compute daylight mean values as well
    matlab_output_vo['j_o3_mean'] = np.mean(matlab_output_vo['j_o3'], axis=2)

    # compute actinic flux in the ozone relevant band (O31P)
    # left boundary should be 193 nm, or even 0, but my calcs start at 200
    # matlab_output_vo['actinic_flux_200_340'] = integrate_spectral_flux(wavelengths, spectral_actinic_flux,
    #                                                                    wl_min=0.200, wl_max=0.34)
    # 220-306 is sensitivity test, this should have most of the qy=0.9
    matlab_output_vo['actinic_flux_220_306'] = integrate_spectral_flux(wavelengths, spectral_actinic_flux,
                                                                       wl_min=0.220, wl_max=0.306)

    # matlab_output_vo['actinic_flux_200_340_mean'] = np.mean(matlab_output_vo['actinic_flux_200_340'], axis=2)
    matlab_output_vo['actinic_flux_220_306_mean'] = np.mean(matlab_output_vo['actinic_flux_220_306'], axis=2)

    # to debug and answer reviewer questions, compute up, down and diff separately
    for key in ['actinic_flux', 'flux_dir_down', 'flux_diff_down', 'flux_up']:
        spectral_flux = matlab_output_vo['spectral_{}'.format(key)][ind]
        matlab_output_vo['{}_200_340'.format(key)] = integrate_spectral_flux(wavelengths, spectral_flux, wl_min=0.200, wl_max=0.34)
        matlab_output_vo['{}_200_340_mean'.format(key)] = np.mean(matlab_output_vo['{}_200_340'.format(key)], axis=2)


def compute_j(wavelengths, spectral_actinic_flux, xs_vo, qy_vo):
    # units are W *m^-2 * um^-1  = J * s^-1 *m^-2 *um^-1
    E = get_photon_energy(wavelengths * 10**-6)
    # this will produce units of photons * cm^-2 * um^-1 * s^-1
    spectral_actinic_flux = spectral_actinic_flux / E[:, np.newaxis, np.newaxis, np.newaxis] * 10**-4

    # abs xsection and q for O2 are rather smooth, interpolate them onto the actinic flux wavelengths
    abs_xsection = np.interp(wavelengths, xs_vo['wavelengths'], xs_vo['abs_xsection'], left=np.NaN, right=np.NaN)
    quantum_yield = np.interp(wavelengths, qy_vo['wavelengths'], qy_vo['quantum_yield'], left=np.NaN, right=np.NaN)

    # units are cm^2 * molecule^-1     *  cm^-2*um^-1^s^-1 =    um^-1*s^-1*molecule^-1
    abs_times_qy = abs_xsection * quantum_yield
    spectral_j = spectral_actinic_flux * abs_times_qy[:, np.newaxis, np.newaxis, np.newaxis]
    # units are s^-1 * molecule^-1
    j = np.trapz(spectral_j, wavelengths, axis=0)
    return spectral_j, j



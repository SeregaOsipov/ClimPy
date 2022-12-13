__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

import numpy as np
from Papers.TobaSuperEruption.modelE.modelE_constants import UVI_ACTINIC_FLUX_SCALE
from climpy.utils.radiation_utils import get_photon_energy

"""
 HOW to compute UV index
 https://www.esrl.noaa.gov/gmd/grad/neubrew/docs/UVindex.pdf
 http://uv.biospherical.com/Version2/Version2.asp
 list of the action spectra
 http://uv.biospherical.com/Version2/description-Version2-Database3.html
 
 need to buy this book
 Human Exposure to Ultraviolet Radiation: Risks and Regulations McKinlay, A. F. and B. L. Diffey (1987)
"""


def get_erythema_action_spectra(wavelength_grid):
    """
    This spectrum is parametrized after A. F. McKinlay and B. L. Diffey (1987).

    for references see:
    http://uv.biospherical.com/Version2/doserates/CIE.txt
    and
    https://www.esrl.noaa.gov/gmd/grad/neubrew/docs/UVindex.pdf

    The input wavelength_grid should be in nm

    :return: action spectra which is used to compute the UV index
    """

    ind = wavelength_grid >= 250
    action_wavelengths = wavelength_grid[ind]
    action_spectrum = np.zeros(action_wavelengths.shape)

    ind = np.logical_and(action_wavelengths >= 250, action_wavelengths <= 298)
    action_spectrum[ind] = 1

    ind = np.logical_and(action_wavelengths > 298, action_wavelengths <= 328)
    action_spectrum[ind] = 10**(0.094*(298-action_wavelengths[ind]))

    ind = np.logical_and(action_wavelengths > 328, action_wavelengths <= 400)
    action_spectrum[ind] = 10**(0.015*(139-action_wavelengths[ind]))

    ind = action_wavelengths > 400
    action_spectrum[ind] = 0

    return action_spectrum, action_wavelengths


def get_DNA_damage_spectra_Setlow_1974(wavelength_grid):
    """
        This spectrum is parametrized after Setlow (194).

        for references see:
        http://uv.biospherical.com/Version2/doserates/SetlowBSI.txt


        The input wavelength_grid should be in nm

        Integration range: 286 - 340 nm

        :return: action spectra that describes the relative biological effect per quantum
        """

    # Integration range: 286 - 340 nm
    ind = np.logical_and(wavelength_grid >= 286, wavelength_grid < 340)
    action_wavelengths = wavelength_grid[ind]
    action_spectrum = np.zeros(action_wavelengths.shape)
    # if np.any(ind):
    #     raise Exception('health_effects_utils:get_erythema_action_spectrum',
    #                     'the wavelength grid is out of the supported range')

    # Figure 1 in the paper by Setlow was parameterized as follows.:
    # A = 10 ^ (13.04679 + (W * -0.047012))	for 286 <= W < 290
    # A = 10 ^ (20.75595 + (W * -0.073595))	for 290 <= W < 295
    # A = 10 ^ (30.12706 + (W * -0.105362))	for 295 <= W < 300
    # A = 10 ^ (42.94028 + (W * -0.148073))	for 300 <= W < 305
    # A = 10 ^ (45.24538 + (W * -0.15563))	for 305 <= W < 340

    params_list = ((286, 290, 13.04679, -0.047012), (290, 295, 20.75595, -0.073595), (295, 300, 30.12706, -0.105362),
                   (300, 305, 42.94028, -0.148073), (305, 340, 45.24538, -0.15563))

    for params in params_list:
        wl_min = params[0]
        wl_max = params[1]
        coef_1 = params[2]
        coef_2 = params[3]

        ind = np.logical_and(action_wavelengths >= wl_min, action_wavelengths < wl_max)
        action_spectrum[ind] = 10 ** (coef_1 + action_wavelengths[ind]*coef_2)

    return action_spectrum, action_wavelengths


def get_DNA_damage_spectra_NDSC(wavelength_grid):
    """
        for references see:
        https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/97JD00072
        Bernhard, Measurements of spectral solar UV irradiance in tropical‐Australia

        The input wavelength_grid should be in nm

        Integration range: ? - 370 nm

        :return: DNA damage action spectra
        """

    action_wavelengths = wavelength_grid
    action_spectrum = np.zeros(action_wavelengths.shape)

    ind = action_wavelengths <= 370
    action_spectrum[ind] = 1/0.0326 * np.exp(13.82 * (1/(1+np.exp( (action_wavelengths[ind]-310)/9 )) - 1))

    ind = action_wavelengths > 370
    action_spectrum[ind] = 0

    return action_spectrum, action_wavelengths


def get_nonmelanoma_skin_cancer_spectra_CIE(wavelength_grid):
    """"
    nmsc_cie.txt from the TUV model
    Action spectrum for the induction of non-melanoma skin cancer. From:
    Photocarcinogenesis Action Spectrum (Non-Melanoma Skin Cancers),
    CIE S 019/E:2006, Commission Internationale de l'Eclairage, 2006.
    1 nm spacing from 250 to 400 nm. Normalized at maximum, 299 nm.
    Set constanta at 3.94E-04 between 340 and 400 nm.
    Wavelength, nm
    """
    file_path = '/home/osipovs/Data/HealthEffects/actionSpectra/TUV_action_spectra/nmsc_cie.txt'
    raw_data = np.loadtxt(file_path, skiprows=7)
    action_wavelengths = raw_data[:, 0]  # nm
    action_spectrum = raw_data[:, 1]  # relative response

    return action_spectrum, action_wavelengths


def get_plants_growth_response_Flint_Caldwell_2003(wavelength_grid):
    #A =EXP(4.688272*EXP(-EXP(0.1703411*(w-307.867)/1.15))+((390-w)/121.7557-4.183832))
    return action_spectrum, action_wavelengths


def compute_uv_index(spectral_irradiance_in, wavelengths_in):
    """
    the UV index should be integrated between 286 and 400 nm
    the range will be extracted from the input data

    see for details
    https://www.esrl.noaa.gov/gmd/grad/neubrew/docs/UVindex.pdf

    :param spectral_irradiance_in: in mW * m^-2 * nm^-1
    :param wavelengths_in: in nm
    :return:
    """

    ind = np.logical_and(wavelengths_in >= 286, wavelengths_in <= 400)
    wavelength_grid = wavelengths_in[ind]  # nm

    spectral_irradiance = spectral_irradiance_in[ind]  # W *m^-2 * um^-1  is equal to mW * m^-2 * nm^-1

    # get the erythemal spectrum
    action_spectrum, action_wavelengths = get_erythema_action_spectra(wavelength_grid)
    uv_index = 1/25 * np.trapz(spectral_irradiance * action_spectrum, wavelength_grid)

    return uv_index, action_spectrum, wavelength_grid


def compute_uv_index_for_ModelE(spectral_flux_in, wavelengths_in):
    """
    see @compute_uv_index for details, this is the version specific for ModelE, because the flux is already integrated within the bin

    Also, the definition of the UV index is changed here, it is based on the actinic flux

    :param spectral_flux_in: ModelE actinic flux output is already integrated dLambda inside the bin,
    which gives the units of [photons * cm^-2 * um^-1 * s^-1] -> [photons * cm^-2 * s^-1]
    :param wavelengths_in: in nm
    :return:
    """

    ind = np.logical_and(wavelengths_in >= 286, wavelengths_in <= 400)
    wavelength_grid = wavelengths_in[ind]  # nm

    E = get_photon_energy(wavelength_grid * 10 ** -9)  # J=W*sec
    # convert photons sec**-1 cm**-2 -> W * m^-2
    spectral_flux = spectral_flux_in[:, ind,:,:] * E[:, np.newaxis, np.newaxis] * 10 ** 4
    spectral_flux = 10 ** 3 * spectral_flux  # W -> mW

    # get the erythemal spectrum
    action_spectrum, action_wavelengths_in = get_erythema_action_spectra(wavelength_grid)
    # since the flux is already integrated, simply weight and sum
    # NOTE extra UVI_ACTINIC_FLUX_SCALE (1/2) scale to match the convention for the background case
    uv_index = UVI_ACTINIC_FLUX_SCALE * 1 / 25 * np.sum(spectral_flux * action_spectrum[:, np.newaxis, np.newaxis], axis=1)

    return uv_index, action_spectrum, wavelength_grid
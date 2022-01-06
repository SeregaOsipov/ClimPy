import numpy as np
import scipy as sp
import scipy.constants

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'


def air_number_density(p, t):
    '''
    Compute air number density using ideal law
    :param p: pressure, [Pa]
    :param t: temperature, [K]
    :return: number density, [molecules m^-3]
    '''

    # Avogadro's number is sp.constants.N_A, 6*10**23, [molecules mol^-1]
    # molar gas constant is sp.constants.R, 8.31, [J mol^-1 K^-1]

    n_a = sp.constants.N_A * p / (sp.constants.R * t)
    return n_a


def air_mass_density(p, t):
    '''
    estimate air mass density, p=rho R T or p = rho * R / M *T, where M is a molar mass

    :param p: pressure, (Pa)
    :param t: temperature, (K)
    :return: air density, (kg/m^3)
    '''

    M = 29 * 10 ** -3  # kg mol^-1
    air_rho = p / sp.constants.R / t * M

    return air_rho


def compute_column_from_vmr_profile(p, t, dz, gas_ppmv, z_dim_axis=0, in_DU=True):
    """
    Computes for a given gas profile the column loading (by default in Dobson Units (DU))
    :param p: in Pa
    :param t: in K (regular,  not potential!)
    :param dz: in meters (derived from z_stag)
    :param gas_ppmv: gas profile in units of ppmv
    :param z_dim_axis:
    :return:
    """

    # TODO: replace these two lines with the decorator
    n_air = air_number_density(p, t)  # molecules / m^3
    gas_number_density = gas_ppmv * 10**-6 * n_air  # molecules / m**3

    gas_dobson_units = compute_column_from_nd_profile(dz, gas_number_density, z_dim_axis, in_DU)
    return gas_dobson_units


def compute_column_from_nd_profile(dz, gas_number_density, z_dim_axis=0, in_DU=True):
    """
    Computes for a given gas profile the column loading (possibly in Dobson Units (DU))
    :param dz: in meters (derived from z_stag)
    :param gas_number_density: gas number density profile in [molecules m^-3]
    :param z_dim_axis:
    :return: gas column (integrated verically) in [molecules m^-2] or in DU
    """

    # dont forget to convert column density from #/m^2 to #/cm^2
    gas_column = np.sum(gas_number_density * dz, axis=z_dim_axis)
    if in_DU:
        DU = 2.69 * 10 ** 20  # molecules m**-2
        gas_column /= DU

    return gas_column
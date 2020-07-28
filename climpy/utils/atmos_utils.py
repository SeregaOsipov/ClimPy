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


def convert_vmr_to_dobson_units(p, t, dz, gas_ppmv, z_dim_axis=0):
    """
    Computes for a given gas profile the column loading in Dobson Units (DU)
    :param p: in Pa
    :param t: in K (regular,  not potential!)
    :param dz: in meters (derived from z_stag)
    :param gas_ppmv:
    :param z_dim_axis:
    :return:
    """

    # TODO: replace these two lines with the decorator
    n_air = air_number_density(p, t)  # molecules / m^3
    gas_number_density = gas_ppmv * 10**-6 * n_air  # molecules / m**3

    gas_dobson_units = convert_nd_to_dobson_units(dz, gas_number_density, z_dim_axis)
    return gas_dobson_units


def convert_nd_to_dobson_units(dz, gas_number_density, z_dim_axis=0):
    """
    Computes for a given gas profile the column loading in Dobson Units (DU)
    :param dz: in meters (derived from z_stag)
    :param gas_number_density: gas number density profile in [molecules m^-3]
    :param z_dim_axis:
    :return:
    """

    DU = 2.69 * 10 ** 20  # molecules m**-2
    # dont forget to convert column density from #/m^2 to #/cm^2
    gas_dobson_units = np.sum(gas_number_density * dz, axis=z_dim_axis) / DU

    return gas_dobson_units
import numpy as np

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'


def compute_gas_dobson_units(p_data, t_data, dz, gas_ppmv, z_dim_axis=0):
    """
    Computes for a given gas profile the column loading in Dobson Units (DU)
    :param p_data: in Pa
    :param t_data: in K (regular,  not potential!)
    :param dz: in meters (derived from z_stag)
    :param gas_ppmv:
    :param z_dim_axis:
    :return:
    """

    avogadro_constant = 6.02 * 10 ** 23  # molecules/mol
    gas_constant = 8.314  # m3 Pa K−1 mol−1
    n_air = avogadro_constant * p_data / gas_constant / t_data  # molecules / m^3

    n_gas_molecules = gas_ppmv * 10 ** -6 * n_air  # molecules / m**3

    # integrate vertically
    # dz = z_stag[1:] - z_stag[:-1]
    DU = 2.69 * 10 ** 20  # molecules m**-2
    # dont forget to convert column density from #/m^2 to #/cm^2
    gas_dobson_units = np.sum(n_gas_molecules * dz, axis=z_dim_axis) / DU

    return gas_dobson_units
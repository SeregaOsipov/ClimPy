__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

import miepython as mp
import numpy as np


def get_mie_efficiencies(ri, r_data, wavelength):
    """

    :param ri:
    :param r_data:
    :param wavelength:
    :return:
    """
    # mp.mie can take x vector as input
    # these are the ext and sca efficiencies, compare to cross sections (qext * A)
    qext = np.empty((len(wavelength), len(r_data)))
    qsca = np.empty((len(wavelength), len(r_data)))

    for wl_index in range(len(wavelength)):
        for r_index in range(len(r_data)):
            m = ri[wl_index]
            # miepython sign convention: imaginary part has to be negative; complex RI = real -j * imaginary
            m = m.real - 1j * m.imag
            x = 2 * np.pi * r_data[r_index] / wavelength[wl_index]
            # efficiencies are without units, area will be in um^2
            qext[wl_index, r_index], qsca[wl_index, r_index], qback, g = mp.mie(m,x)

    mie_vo = {}
    mie_vo['qext'] = qext
    mie_vo['qsca'] = qsca
    mie_vo['wavelength'] = wavelength
    mie_vo['r_data'] = r_data
    return mie_vo
import miepython as mp
import numpy as np

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'


def get_mie_efficiencies(ri, r_data, wavelength):
    """
    ri = ri_vo['ri'][ind]
    r_data = dA_vo['radii']
    wavelength = ri_vo['wl']

    :param ri:
    :param r_data: units should be the same as wavelength
    :param wavelength: units should be the same as radius
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
            m = m.real - 1j * np.abs(m.imag)
            x = 2 * np.pi * r_data[r_index] / wavelength[wl_index]
            # efficiencies are without units, area will be in um^2
            qext[wl_index, r_index], qsca[wl_index, r_index], qback, g = mp.mie(m,x)

    # dummy checks
    if np.any(qext < 0):
        raise Exception('Mie solution can not have negative extinction values, check RI')

    mie_vo = {}
    mie_vo['qext'] = qext
    mie_vo['qsca'] = qsca
    mie_vo['wavelength'] = wavelength
    mie_vo['r_data'] = r_data
    return mie_vo


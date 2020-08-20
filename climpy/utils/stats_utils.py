import numpy as np

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'


def get_cdf(f, radii):
    """
    Compute cdf: integral of f(x)*dx up to x
    :param f:
    :param radii:
    :return:
    """
    cdf = []
    for r in radii:
        ind = radii <= r
        cdf.append(np.trapz(f[ind], radii[ind]))
    return np.array(cdf)
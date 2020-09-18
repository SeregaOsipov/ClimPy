import numpy as np
import scipy as sp
from climpy.utils.diag_decorators import normalize_size_distribution_by_area

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'


@normalize_size_distribution_by_area
def get_Kok_dust_emitted_size_distribution(moment='dN'):
    # Kok et. al, 2011, equations 5 and 6
    cn = 0.9539  # m
    cv = 12.62  # m
    ds = 3.4  # m
    ss = 3.0
    lambd = 12  # m

    dp = np.logspace(-9, -4, 40)  # m
    dd = dp * 10 ** 6

    dNdlogd = cn ** -1 * dd ** -2 * (1 + sp.special.erf(np.log(dd / ds) / (2 ** 0.5 * np.log(ss)))) * np.exp(-(dd / lambd) ** 3)
    dVdlogd = cv ** -1 * dd * (1 + sp.special.erf(np.log(dd / ds) / (2 ** 0.5 * np.log(ss)))) * np.exp(-(dd / lambd) ** 3)

    dNdlogd_vo = {}
    dNdlogd_vo['data'] = dNdlogd
    dNdlogd_vo['radii'] = dp/2 * 10**6  # um

    dVdlogd_vo = {}
    dVdlogd_vo['data'] = dVdlogd
    dVdlogd_vo['radii'] = dp / 2 * 10 ** 6  # um

    # TODO: Kok PSD are not exactly 1 in [0.2, 20]

    # check normalization, should be 1
    ind = np.logical_and(dp >= 0.2 * 10 ** -6, dp <= 20 * 10 ** -6)
    logdd = np.log(dd)
    print('Kok dNdlogd [0.2-20] area is {}'.format(np.trapz(dNdlogd[ind], logdd[ind])))
    # np.trapz(dVdlogd, logdd)
    print('Kok dVdlogd [0.2-20] area is {}'.format(np.trapz(dVdlogd[ind], logdd[ind])))

    vo = dNdlogd_vo
    if moment is 'dV':
        vo = dVdlogd_vo
    return vo
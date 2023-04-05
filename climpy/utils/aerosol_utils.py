import numpy as np
import scipy as sp

from climpy.utils import mie_utils as mie
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
    if moment == 'dV':
        vo = dVdlogd_vo
    return vo


def derive_aerosols_optical_properties(ri_vo, dA_vo):
    '''
    Use this for a single aerosols type and loop through the list
    Currently only extinction / optical depth

    :param ri_vo: RI of the aerosols
    :param dA_vo: cross-section area distribution
    :return:
    '''

    # ri_wl = ri_vo['wl']
    # qext = np.zeros(dA_vo['data'].shape)
    # with np.nditer(qext, op_flags=['readwrite']) as it_q:
    #     with np.nditer(ri_vo['ri']) as it_ri:
    #         for q, ri in zip(it_q, it_ri):
    #             print(q, ri, ri_wl)
    #             # mie_vo = mie.get_mie_efficiencies(ri, dA_vo['radii'], ri_wl)
    #             mie_vo = mie.get_mie_efficiencies(ri[np.newaxis], dA_vo['radii'], ri_wl)
    #             q[...] = np.squeeze(mie_vo['qext'])

    # Compute Mie extinction coefficients
    # dims are time, r, wl
    qext = np.zeros(dA_vo['data'].shape + ri_vo['wl'].shape)
    for time_index in range(qext.shape[0]):
        # debug
        ri, r_data, wavelength = ri_vo['ri'][time_index], dA_vo['radii'], ri_vo['wl']
        mie_vo = mie.get_mie_efficiencies(ri_vo['ri'][time_index], dA_vo['radii'], ri_vo['wl'])
        qext[time_index] = np.swapaxes(mie_vo['qext'], 0, 1)

    # dims: time, r, wl & time, r
    integrand = qext * dA_vo['data'][..., np.newaxis]
    column_od = np.trapz(integrand, np.log(dA_vo['radii']), axis=1)  # sd is already dAdlnr
    # column_od = np.sum(column_od_by_modes, axis=1)  # sum up modes

    return column_od



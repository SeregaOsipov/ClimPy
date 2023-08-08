import numpy as np
import scipy as sp
import xarray as xr
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
    dNdlogd_vo['radius'] = dp/2 * 10**6  # um

    dVdlogd_vo = {}
    dVdlogd_vo['data'] = dVdlogd
    dVdlogd_vo['radius'] = dp / 2 * 10 ** 6  # um

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


def derive_aerosols_optical_properties(ri_ds, dA_ds):
    '''
    Use this for a single aerosols type and loop through the list
    Currently only extinction / optical depth

    :param ri_ds: RI of the aerosols
    :param dA_ds: cross-section area distribution
    :return:
    '''

    # ri_wl = ri_vo['wl']
    # qext = np.zeros(dA_vo['data'].shape)
    # with np.nditer(qext, op_flags=['readwrite']) as it_q:
    #     with np.nditer(ri_vo['ri']) as it_ri:
    #         for q, ri in zip(it_q, it_ri):
    #             print(q, ri, ri_wl)
    #             # mie_ds = mie.get_mie_efficiencies(ri, dA_vo['radius'], ri_wl)
    #             mie_ds = mie.get_mie_efficiencies(ri[np.newaxis], dA_vo['radius'], ri_wl)
    #             q[...] = np.squeeze(mie_ds['qext'])

    # Compute Mie extinction coefficients
    # dims are time, r, wl
    qext_ds = np.zeros(dA_ds['dAdlogd'].shape + ri_ds['wavelength'].shape)
    qexts = []
    for time_index in range(dA_ds.time.size):
        mie_ds = mie.get_mie_efficiencies(ri_ds.ri.isel(time=time_index), dA_ds['radius'], ri_ds.wavelength)
        qexts.append(mie_ds.qext)
    qext_ds = xr.concat(qexts, dim='time')

    # dims: time, r, wl & time, r
    integrand = qext_ds * dA_ds['dAdlogd']
    column_od = np.trapz(integrand, np.log(dA_ds['radius']), axis=integrand.dims.index('radius'))  # sd is already dAdlnr
    # column_od = np.trapz(integrand, np.log(dA_ds['radius']), axis=1)  # sd is already dAdlnr
    # column_od = np.sum(column_od_by_modes, axis=1)  # sum up modes

    return column_od



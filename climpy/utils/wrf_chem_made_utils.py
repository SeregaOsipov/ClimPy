import numpy as np
import xarray as xr
import pandas as pd
import scipy as sp
import scipy.special

from climpy.utils.diag_decorators import time_interval_selection, normalize_size_distribution_by_point, derive_size_distribution_moment
from climpy.utils.netcdf_utils import convert_time_data_impl
from climpy.utils.wrf_chem_utils import get_aerosols_keys, get_molecule_key_from_aerosol_key, to_stp, combine_aerosol_types, \
    combine_aerosol_modes

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

"""
MADE specific utils
"""


# !  initial mean diameter for nuclei mode, accumulation and coarse modes [ m ]
MADE_MODES_DGs = [0.01E-6, 0.07E-6, 1.0E-6]
# !  initial sigma-G for nuclei, acc, and coarse modes
MADE_MODES_SIGMA = [1.7, 2.0, 2.5]  #       PARAMETER (sginin=1.70)

# this one was ported from MADE/VBS module_data_soa_vbs.F
# these are the values used in chem_opt=100 (used in the *fac, i.e. orgfac or no3fac)
# USE THIS ONE
MADE_VBS_AEROSOLS_DENSITY_MAP = {  # component densities [ kg/m**3 ] :
    'so4':1.8E3,
    'nh4':1.8E3,
    'no3':1.8E3,
    'h2o':1.0E3,
    'org':1.0E3,
    'soil':2.6E3,
    'seas':2.2E3,
    'anth':2.2E3,
    'na':2.2E3,
    'cl':2.2E3,
    'ca':2.6E3,
    'k':2.6E3,
    'mg':2.6E3,

    # manual
    'oc':1.5E3,
    'ec':2.2E3,  # anth

    # primary organics = just org
    'orgp':1.0E3,  # org
    'p25':2.2E3,  # anth

    # manually add VBS SOA aerosols
    # anthropogenic
    'asoa1':1.0E3,  # org
    'asoa2':1.0E3,
    'asoa3':1.0E3,
    'asoa4':1.0E3,
    # biogenic
    'bsoa1':1.0E3,  # org
    'bsoa2':1.0E3,
    'bsoa3':1.0E3,
    'bsoa4':1.0E3,
}

# this one also was ported from MADE/VBS module_data_soa_vbs.F
# this values are NOT used in chem_opt=100,
# fortran vars are called dens_type_aer, i.e. dens_so4_aer, search for dens_mastercomp_aer
# DO NOT USE THIS ONE
MADE_VBS_AEROSOLS_DENSITY_MAP_dens_mastercomp_aer = {  # component densities [ kg/m**3 ] :
    'so4':1.8E3,
    'nh4':1.8E3,
    'no3':1.8E3,
    'h2o':1.0E3,
    'org':1.0E3,
    'soil':2.6E3,
    'seas':2.2E3,
    'anth':2.2E3,  # oin
    'na':2.2E3,
    'cl':2.2E3,
    'ca':2.6E3,
    'k':2.6E3,
    'mg':2.6E3,

    # manual
    'oc':1.5E3,
    'ec':1.7E3,

    # primary organics = just org
    'orgp':1.0E3,  # oc
    'p25':2.2E3,  # oin

    # manually add VBS SOA aerosols
    # anthropogenic
    'asoa1':1.5E3,  # same as OC
    'asoa2':1.5E3,
    'asoa3':1.5E3,
    'asoa4':1.5E3,
    # biogenic
    'bsoa1':1.5E3,  # same as OC
    'bsoa2':1.5E3,
    'bsoa3':1.5E3,
    'bsoa4':1.5E3,
}

# WRF SW RIs maped to the chem_opt 100, from module_data_rrtmgaeropt.F
# (nswbands =4,nlwbands=16)
# MADE_VBS_AEROSOLS_SW_REFRACTIVE_INDEX_MAP = {  # complex refractive indicies
#     'so4':1.8E3,
#     'nh4':1.8E3,
#     'no3':1.8E3,
#     'h2o':1.0E3,
#     'org':1.0E3,
#     'soil':2.6E3,
#     'seas':2.2E3,
#     'anth':2.2E3,
#     'na':2.2E3,
#     'cl':2.2E3,
#     'ca':2.6E3,
#     'k':2.6E3,
#     'mg':2.6E3,
#
#     # manual
#     'oc':1.5E3,
#     'ec':2.2E3,  # anth
#
#     # primary organics = just org
#     'orgp':1.0E3,  # org
#     'p25':2.2E3,  # anth
#
#     # manually add VBS SOA aerosols
#     # anthropogenic
#     'asoa1':1.0E3,  # org
#     'asoa2':1.0E3,
#     'asoa3':1.0E3,
#     'asoa4':1.0E3,
#     # biogenic
#     'bsoa1':1.0E3,  # org
#     'bsoa2':1.0E3,
#     'bsoa3':1.0E3,
#     'bsoa4':1.0E3,
# }


def get_WRF_MADE_modpar(sgs, m0s, m3s):
    """
    see ACKERMANN et al., 1998 MODAL AEROSOL DYNAMICS MODEL FOR EUROPE: DEVELOPMENT AND FIRST APPLICATIONS
    https://www.sciencedirect.com/science/article/abs/pii/S1352231098000065

    code was ported from WRF Chem module_aerosols_sorgam.f (and data)

    :param m0s = (nu0, ac0, cor0)  # 0 moments
    :param m3s = (nu3, ac3, cor3)  # 3rd moments
    :param sgs = (1.7, 2.0, 2.5)  # sginin, sginia, sginic # initial sigma-G for nuclei, acc, coarse modes

    :return: median diameters computed as:
    dgacc(lcell) = max(dgmin,(cblk(lcell,vac3)/(cblk(lcell,vac0)*esa36))**one3)
    """

    # TODO add cw_phase logic, see the source code
    dgs = ()
    dgmin = 1.0E-09
    for sg, m3, m0 in zip(sgs, m3s, m0s):
        es36 = np.exp(0.125 * np.log(sg) ** 2) ** 36
        dg = (m3 / (m0 * es36)) ** (1 / 3)
        if isinstance(dg, np.ndarray):
            dg[dg < dgmin] = dgmin  # numpy implementation
        else:  # xarray
            dg = dg.where(dg >= dgmin, dgmin)  # replace values < dgmin with dgmin
        dgs += (dg,)

    # sginin = 1.70
    # esn36 = np.exp(0.125*np.log(sginin) ** 2) ** 36
    # dgacc = (ac3/ac0*esn36)**(1/3)
    # dgmin = 1.0E-09
    # dgacc[dgacc < dgmin] = dgmin

    return dgs


def get_wrf_sd_params(xr_in):
    """
    m0 units are [# m^-3]
    m3 units are [m^3 m^-3]
    """
    m0s = [xr_in['nu0'], xr_in['ac0'], xr_in['corn']]  # 0 moments
    m3s = [xr_in['NU3'], xr_in['AC3'], xr_in['COR3']]  # 3rd moments

    # Note: that inverse density (ALT) is only applied to aerosols (to and convert from mixing ratio x/kg-dryair to x/m^3) and not to moments m0s and m3s
    # inv_density = nc['ALT'][:]
    # m0s = [moment / inv_density for moment in m0s]
    # m3s = [moment / inv_density for moment in m3s]

    sgs = [np.ones(m0s[0].shape)*sg for sg in MADE_MODES_SIGMA]
    sgs = [xr.DataArray(data=sg, coords=m0s[0].coords, dims=m0s[0].dims, name='sg{}'.format(index)) for index, sg in enumerate(sgs)]  # convert to xarray

    # derive the median diameter
    dgs = get_WRF_MADE_modpar(sgs, m0s, m3s)

    # check bad values
    for m0, m3 in zip(m0s, m3s):
        ind = np.isnan(m0+m3)
        if np.sum(ind) > 0:
            raise Exception('m0 or m3 has {} nan values and there should none'.format(np.sum(ind)))

    # convert to xarray structure
    m0s = xr.concat(m0s, dim='mode').rename('m0s')
    m3s = xr.concat(m3s, dim='mode').rename('m3s')
    sgs = xr.concat(sgs, dim='mode').rename('sgs')
    dgs = xr.concat(dgs, dim='mode').rename('dgs')

    sd_ds = xr.merge([sgs, dgs, m0s, m3s])
    sd_ds['mode'] = ['i', 'j', 'k']

    # return sgs, dgs, m0s, m3s
    return sd_ds


def derive_m3s_from_mass_concentrations(xr_in, chem_opt, wet=False, sum_up_components=True):
    """
    Derive m3s from individual output of aerosols (like it is done in module_optical_averaging.F)
    return m3s [m^3 m^-3]
    """

    # get densities array in the order of the keys
    aerosols_keys = get_aerosols_keys(chem_opt, wet=wet)
    aerosols_densities = []
    for key in aerosols_keys:
        i = aerosols_keys.index(key)
        rho_key = get_molecule_key_from_aerosol_key(key)
        aerosols_densities += (MADE_VBS_AEROSOLS_DENSITY_MAP[rho_key],)
    aerosols_densities = np.stack(aerosols_densities)  # kg * m^-3

    aerosols_masses, dummy = get_aerosols_pm_stack(xr_in, aerosols_keys)  # ug m**-3
    aerosols_volumes_by_type = (aerosols_masses.T / aerosols_densities).T * 10 ** -9  # m**3

    aerosols_volumes = aerosols_volumes_by_type
    if sum_up_components:
        aerosols_volumes = aerosols_volumes_by_type.sum(axis=0)  # sum up individual components

    m3_pp = aerosols_volumes * 6 / np.pi  #  Volume is: pi/6 * d**3

    return m3_pp


def sample_WRF_MADE_size_distributions(sd_ds): # dp, sgs, dgs, m0s, m3s):
    """
    All parameters are by mode (list for each mode)
    :param dp:
    :param sgs: list of sg by mode
    :param dgs:
    :param m3s: list of 3rd momemnts
    :param m0s:
    :return: the dNdlog(p) size distribution for each mode and scaled to total number of particles
    """

    sd_ds['dNdlogd'] = 1 / ((2 * np.pi) ** (1 / 2) * np.log(sd_ds['sgs'])) * np.exp(-1 / 2 * (np.log(sd_ds['radius']*2) - np.log(sd_ds['dgs'])) ** 2 / np.log(sd_ds['sgs']) ** 2)

    # dNdlogds = ()  # list of dNdlogd by mode
    # for sg, dg, m0, m3 in zip(sgs, dgs, m0s, m3s):
    #     if isinstance(sg, float):  # # if sg/dg... are numbers, then convert them to arrays
    #         sg = np.array(sg)
    #         dg = np.array(dg)
    #         m0 = np.array(m0)
    #         m3 = np.array(m3)
    #
    #     # THIS one is faster then sp.stats.lognorm
    #     dNdlogd = 1/((2*np.pi)**(1/2) * np.log(sg[..., np.newaxis])) * np.exp(-1/2 * (np.log(dp)-np.log(dg[..., np.newaxis]))**2 / np.log(sg[..., np.newaxis])**2)
    #     # lognorm_dist = sp.stats.lognorm(s=np.log(sg[..., np.newaxis]), loc=0, scale=dg[..., np.newaxis])
    #     # dNdlogp = dp * lognorm_dist.pdf(dp)
    #     dNdlogd *= m0[..., np.newaxis]  # this is the n(logdp) from ACKERMANN et al., equation 1
    #     dNdlogds += (dNdlogd,)
    # return dNdlogds

    return sd_ds

def compute_MADE_bounded_distribution_factors(d_min, d_max, sgs, dgs, m3s, m0s):
    """
    Computes the Number and Volume factors, that represent the contribution of the size distribution
    to the total number of particles and volume for a given range of size.

    :param d_min:
    :param d_max:
    :return: Returns 2 factors for number of particles and volume.
    1st factor to multiply total mass and to get PM, i.e. PM[d>=d_min, d<=d_max] 10 = Total mass * factor
    2nd factor to multiply total number of particles and to get N[d>=d_min, d<=d_max] i.e. N[d>min, d<max] = N_0 * factor
    """

    # this factors are the ratio that they contribute to the total number N_0 or volume V_0
    N_factors = []
    V_factors = []
    for sg, dg, m3, m0 in zip(sgs, dgs, m3s, m0s):
        d_v_median = dg * np.exp(3 * np.log(sg) ** 2)  # volume median diameter

        # See for details https://patarnott.com/pdf/SizeDistributions.pdf, eqn 24-27
        # N[Dmin, Dmax] = N_0/2 * []
        N_factor = 1 / 2 * (sp.special.erf(np.log(d_max / dg) / (2 ** 0.5 * np.log(sg))) - sp.special.erf(
            np.log(d_min / dg) / (2 ** 0.5 * np.log(sg))))
        # V[Dmin, Dmax] = V_0/2 * []   ! There is typo in equation 27, it should be V_0, not N_0
        V_factor = 1 / 2 * (sp.special.erf(np.log(d_max / d_v_median) / (2 ** 0.5 * np.log(sg))) - sp.special.erf(
            np.log(d_min / d_v_median) / (2 ** 0.5 * np.log(sg))))

        # V_0 = N_0  * pi/6 * dg^3 * exp(9/2 ln(sg)^2)
        # NtoV_factor = (np.pi / 6 * dg ** 3 * np.exp(9 / 2 * np.log(sg) ** 2)) ** -1

        N_factors.append(N_factor)
        V_factors.append(V_factor)

    return N_factors, V_factors


def get_WRF_MADE_initial_params(m3s):
    '''
    Compute m0 given m3 using default dg and sg
    For example, default m3s for dust in WRF is

    :return: the default parameters of the initial size distributions given by MADE in WRF-Chem
    '''

    dgs = MADE_MODES_DGs
    sgs = MADE_MODES_SIGMA

    m0s = []  # derived from m3s
    # m3s = [0, 7 / 100, 93 / 100]  # example for dust

    for sg, dg, m3 in zip(sgs, dgs, m3s):
        es36 = np.exp(0.125 * np.log(sg) ** 2) ** 36
        # chem(i,k,j,p_ac0) = m3acc/((dginia**3)*esa36)
        m0 = m3 / ((dg**3) * es36)
        m0s.append(m0)

    return sgs, dgs, m0s, m3s


def define_MADE_modes_by_aerosol_type(aerosols_keys):
    """
    MADE (WRF-Chem) defines 3 modes: Aitken, Accumulation and Coarse
    This modes are index 0, 1 and 2.

    This routine assigns each aerosol type to a mode index

    Routine is used to compute PM V factors across different species

    :param aerosols_keys:
    :return: return key(aerosol type)-index dictionary
    """

    mode_indices = {}
    for key in aerosols_keys:
        # NOTE: this logic wil break, if the var names change
        if key[-1] == 'i':
            mode_indices[key] = 0
        elif key[-1] == 'j':
            mode_indices[key] = 1
        else:  # antha, seas, soila
            mode_indices[key] = 2

    return mode_indices


@to_stp
@combine_aerosol_types
@combine_aerosol_modes
def get_aerosols_pm_stack(xr_in, aerosols_keys, pm_size_range=None, pm_input_is_aerodynamic_diameter=True):
    """
    Computes the aerosols PM mass concentration. Note the difference between different diameters: aerodynamic, geometric, otpical, etc.
    AQ stations measure PM2.5 in aerodynamic diameter.

    In WRF, we always work with geom diameters.

    Examples:
    To compute PM for [d>250nm and < 10um] set pm_sizes = [0.25 * 10 ** -6, 10 * 10 ** -6]  # m

    :param xr_in: netcdf as xarr
    :param aerosols_keys:
    :param pm_size_range: [d_min, d_max] in meters, geometric diameter
    :param pm_input_is_aerodynamic_diameter: if true, pm_size_range is specified in aerodynamic diamter, else in geometric.
    :return:
    """

    if pm_input_is_aerodynamic_diameter:  # convert aerodynamic to geometric diameter, prepare the densities
        aerosol_densities_df = pd.DataFrame.from_dict(MADE_VBS_AEROSOLS_DENSITY_MAP, orient='index', columns=['rho', ]) * 10**-3  # convert kg/m^3 to g/cm^3

    # get the mode index for each aerosol type
    MADE_mode_indices = define_MADE_modes_by_aerosol_type(aerosols_keys)

    diags = []
    alt = xr_in.variables['ALT'][:]
    for key in aerosols_keys:
        # all vars are [ug/kg-dryair], alt is [m3 kg-1]
        # print('{} [{}]'.format(key, nc.variables[key].units))
        diag = xr_in[key][:] / alt

        # by default assign V factors to 1 for all types, it is equal to PM from 0 to Infinity
        default = np.ones(xr_in.variables['ALT'].shape)
        V_factors = [default, default, default]  # array of ones
        if pm_size_range is not None:  # then compute the V factors for each aerosol type
            sgs, dgs, m0s, m3s = get_wrf_sd_params(xr_in)
            d_min = pm_size_range[0]
            d_max = pm_size_range[1]
            # I have to compute the V factor for each aerosol type, because conversation to aerodynamic diameter involves density
            if pm_input_is_aerodynamic_diameter:  # d_aer = rho**0.5 * d_geom  # else keep in geometric
                rho_key = get_molecule_key_from_aerosol_key(key)
                rho = aerosol_densities_df.loc[rho_key].rho
                d_min /= rho ** 0.5  # this still will be zero
                d_max /= rho ** 0.5  # this will convert aerodynamic diameter to geometric
            N_factors, V_factors = compute_MADE_bounded_distribution_factors(d_min, d_max, sgs, dgs, m3s, m0s)  # this input always requires geometric diameters

        # apply the PM factor (size correction)
        mode_index = MADE_mode_indices[key]
        diag *= V_factors[mode_index]

        diags.append(diag)  # [ug m**-3]

    # after reading convert all keys to lower case
    aerosols_keys = tuple(key.lower() for key in aerosols_keys)

    # combine into dataset merging via new dim
    pm_ds = xr.concat(diags, 'aerosol').rename('pm_by_type')
    pm_ds['aerosol'] = list(aerosols_keys)

    return pm_ds, aerosols_keys


def rank_aerosols_contribution_to_the_mode(nc, aerosols_keys):
    '''
    This routine compute the contribution of the aerosol type to the size mode
    :param nc:
    :param aerosols_keys:
    :return:
    '''
    diags_vstack, keys = get_aerosols_pm_stack(nc, aerosols_keys, combine_organics=True, combine_other=True)

    # integrate vertically
    z_stag = (nc['PH'] + nc['PHB']) / 9.8
    z_stag = np.squeeze(z_stag)
    dz = np.diff(z_stag, axis=1)  # m

    diags_vstack = np.sum(diags_vstack * dz, axis=2)

    # time mean
    diags_vstack = np.mean(diags_vstack, axis=1)

    # sort according to contribution
    ind = np.argsort(diags_vstack)
    # Reverse the sorted array
    ind = ind[::-1]

    # ind_3d = np.repeat(ind[:, np.newaxis, :], diags_vstack.shape[1], axis=1)
    # np.take_along_axis(diags_vstack, ind_3d, axis=1).shape

    return diags_vstack[ind], np.array(keys)[ind].tolist()


@time_interval_selection
# @geo_regions_time_averaging
@normalize_size_distribution_by_point
@derive_size_distribution_moment
def get_wrf_size_distribution_by_modes(xr_in, sum_up_modes=False, column=False, r_grid_to_merge=None, derive_m3=False, chem_opt=None):  # , wet=True
    '''
    :param xr_in:
    :param r_grid_to_merge: additional sampling points, [m]
    :return: original dNdlogr has units [part / m^3], this can be modified by
    decorator derive_size_distribution_moment
    '''

    sd_ds = get_wrf_sd_params(xr_in)  # size distribution parameters: sgs, dgs, m0s, m3s
    if derive_m3:  # derive m3 from individual components instead of the direct output
        m3s = derive_m3s_from_mass_concentrations(xr_in, chem_opt, wet=False)
        sd_ds['m3s'] = xr.concat(m3s, dim='mode').rename('m3s')

    dp = np.logspace(-9, -4, 40)  # sample the distribution  # dp = np.logspace(-9, -4, 100)
    # TODO: move these to r_grid_to_merge
    # add Aeronet and Drewnick radius for normalization
    # dp = np.append(dp, 2*AERONET_NORMALIZATION_RADUIS * 10 ** -6)
    # dp = np.append(dp, 2*DREWNICK_NORMALIZATION_RADUIS * 10 ** -6)
    if r_grid_to_merge is not None:
        dp = np.append(dp, r_grid_to_merge*2)
    dp.sort()

    sd_ds['radius'] = dp / 2
    sd_ds.radius.attrs['units'] = 'm'

    sd_ds = sample_WRF_MADE_size_distributions(sd_ds) #dp, sgs, dgs, m0s, m3s)
    sd_ds['radius'] = sd_ds['radius'] * 10**6  # um
    sd_ds.radius.attrs['units'] = 'um'

    # Aux stuff
    if sum_up_modes:
        sd_ds['dNdlogd'] = sd_ds['dNdlogd'].sum(dim='mode')

    if column:  # integrate vertically
        z_stag = (xr_in['PH'] + xr_in['PHB']) / 9.8
        dz = z_stag.diff(dim='bottom_top_stag').rename({'bottom_top_stag': 'bottom_top'})
        sd_ds['dNdlogd'] = (sd_ds['dNdlogd'] * dz).sum(dim='bottom_top')
        # TODO: do not change units here
        sd_ds['dNdlogd'] *= 10 ** -12  # particles * um**3 / um**2  # convert units: WRF [part * um^3 / m^3 * m] to Aeronet [um^3/um^2]

        # z_dim = vo['data'].shape.index(xr_in.dimensions['bottom_top'].size)  # deduce z_dim index
        # vo['data'] = np.sum(vo['data'] * dz[..., np.newaxis], axis=z_dim)
        # vo['data'] *= 10 ** -12  # particles * um**3 / um**2  # convert units: WRF [part * um^3 / m^3 * m] to Aeronet [um^3/um^2]

    return sd_ds
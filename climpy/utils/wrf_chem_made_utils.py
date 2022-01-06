import numpy as np
import scipy as sp
import scipy.special

from climpy.utils.diag_decorators import time_interval_selection, normalize_size_distribution_by_point, derive_size_distribution_moment
from climpy.utils.netcdf_utils import convert_time_data_impl
from climpy.utils.wrf_chem_utils import get_aerosols_keys, get_molecule_key_from_aerosol_key, to_stp, vstack_and_sort_aerosols, combine_aerosol_types, \
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
# this are the values used in chem_opt=100 (used in the *fac, i.e. orgfac or no3fac)
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
        dg[dg < dgmin] = dgmin
        dgs += (dg,)

    # sginin = 1.70
    # esn36 = np.exp(0.125*np.log(sginin) ** 2) ** 36
    # dgacc = (ac3/ac0*esn36)**(1/3)
    # dgmin = 1.0E-09
    # dgacc[dgacc < dgmin] = dgmin

    return dgs


def get_wrf_sd_params(nc):
    """
    m0 units are [# m^-3]
    m3 units are [m^3 m^-3]
    """
    m0s = [nc['nu0'][:], nc['ac0'][:], nc['corn'][:]]  # 0 moments
    m3s = [nc['NU3'][:], nc['AC3'][:], nc['COR3'][:]]  # 3rd moments

    # Note: that inverse density (ALT) is only applied to aerosols (to and convert from mixing ratio x/kg-dryair to x/m^3) and not to moments m0s and m3s
    # inv_density = nc['ALT'][:]
    # m0s = [moment / inv_density for moment in m0s]
    # m3s = [moment / inv_density for moment in m3s]

    sgs = [np.ones(m0s[0].shape)*sg for sg in MADE_MODES_SIGMA]
    # derive the median diameter
    dgs = get_WRF_MADE_modpar(sgs, m0s, m3s)

    # check bad values
    for m0, m3 in zip(m0s, m3s):
        ind = np.isnan(m0+m3)
        if np.sum(ind) > 0:
            raise Exception('m0 or m3 has {} nan values and there should none'.format(np.sum(ind)))

    # if sgs[0].ndim > 3:  # due to memory restrictions, convert float64 to float16
    #     sgs = [item.astype(np.float16) for item in sgs]
    #     dgs = [item.astype(np.float16) for item in dgs]
    #     m0s = [item.astype(np.float16) for item in m0s]
    #     m3s = [item.astype(np.float16) for item in m3s]

    return sgs, dgs, m0s, m3s


def derive_m3s_from_mass_concentrations(nc, chem_opt, wet=False, sum_up_components=True):
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

    aerosols_masses, dummy = get_aerosols_stack(nc, aerosols_keys)  # ug m**-3
    aerosols_volumes_by_type = (aerosols_masses.T / aerosols_densities).T * 10 ** -9  # m**3

    aerosols_volumes = aerosols_volumes_by_type
    if sum_up_components:
        aerosols_volumes = aerosols_volumes_by_type.sum(axis=0)  # sum up individual components

    m3_pp = aerosols_volumes * 6 / np.pi  #  Volume is: pi/6 * d**3

    return m3_pp


def sample_WRF_MADE_size_distributions(dp, sgs, dgs, m0s, m3s):
    """
    All parameters are by mode (list for each mode)
    :param dp:
    :param sgs: list of sg by mode
    :param dgs:
    :param m3s: list of 3rd momemnts
    :param m0s:
    :return: the dNdlog(p) size distribution for each mode and scaled to total number of particles
    """

    dNdlogds = ()  # list of dNdlogd by mode
    for sg, dg, m0, m3 in zip(sgs, dgs, m0s, m3s):
        if isinstance(sg, float):  # # if sg/dg... are numbers, then convert them to arrays
            sg = np.array(sg)
            dg = np.array(dg)
            m0 = np.array(m0)
            m3 = np.array(m3)

        # THIS one is faster then sp.stats.lognorm
        dNdlogd = 1/((2*np.pi)**(1/2) * np.log(sg[..., np.newaxis])) * np.exp(-1/2 * (np.log(dp)-np.log(dg[..., np.newaxis]))**2 / np.log(sg[..., np.newaxis])**2)
        # lognorm_dist = sp.stats.lognorm(s=np.log(sg[..., np.newaxis]), loc=0, scale=dg[..., np.newaxis])
        # dNdlogp = dp * lognorm_dist.pdf(dp)
        dNdlogd *= m0[..., np.newaxis]  # this is the n(logdp) from ACKERMANN et al., equation 1
        dNdlogds += (dNdlogd,)

    return dNdlogds


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
@vstack_and_sort_aerosols
@combine_aerosol_types
@combine_aerosol_modes
def get_aerosols_stack(nc, aerosols_keys, pm_sizes=None):
    """
    Computes the aerosols PM mass concentration

    Examples:
    To compute PM for [d>250nm and < 10um] set pm_sizes = [0.25 * 10 ** -6, 10 * 10 ** -6]  # m

    :param nc:
    :param aerosols_keys:
    :param pm_sizes: [d_min, d_max] in meters,
    :return:
    """

    # by default assign V factors to 1 for all types, it is equal to PM from 0 to Infinity
    default = np.ones(nc.variables['ALT'].shape)
    V_factors = [default, default, default]

    if pm_sizes is not None:  # then compute the V factors for each aerosol type
        sgs, dgs, m0s, m3s = get_wrf_sd_params(nc)
        d_min = pm_sizes[0]
        d_max = pm_sizes[1]
        N_factors, V_factors = compute_MADE_bounded_distribution_factors(d_min, d_max, sgs, dgs, m3s, m0s)

    # get the mode index for each aerosol type
    MADE_mode_indices = define_MADE_modes_by_aerosol_type(aerosols_keys)

    diags = []
    alt = nc.variables['ALT'][:]
    for key in aerosols_keys:
        # all vars are [ug/kg-dryair], alt is [m3 kg-1]
        # print('{} [{}]'.format(key, nc.variables[key].units))
        diag = nc.variables[key][:] / alt

        # apply the PM factor (size correction)
        mode_index = MADE_mode_indices[key]
        diag *= V_factors[mode_index]

        diags.append(diag)  # [ug m**-3]  # np.squeeze(

    # after reading convert all keys to lower case
    aerosols_keys = tuple(key.lower() for key in aerosols_keys)

    return diags, aerosols_keys


def rank_aerosols_contribution_to_the_mode(nc, aerosols_keys):
    '''
    This routine compute the contribution of the aerosol type to the size mode
    :param nc:
    :param aerosols_keys:
    :return:
    '''
    diags_vstack, keys = get_aerosols_stack(nc, aerosols_keys, combine_organics=True, combine_other=True)

    # integrate vertically
    z_stag = nc['PH'][:] + nc['PHB'][:] / 9.8
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
def get_wrf_size_distribution_by_modes(nc, sum_up_modes=False, column=False, r_grid_to_merge=None, derive_m3=False, chem_opt=None):  # , wet=True
    '''
    :param nc:
    :param r_grid_to_merge: additional sampling points, [m]
    :return: original dNdlogr has units [part / m^3], this can be modified by
    decorator derive_size_distribution_moment
    '''

    sgs, dgs, m0s, m3s = get_wrf_sd_params(nc)  # size distribution parameters
    if derive_m3:  # derive m3 from individual components instead of the direct output
        m3s = derive_m3s_from_mass_concentrations(nc, chem_opt, wet=False)

    dp = np.logspace(-9, -4, 40)  # sample the distribution  # dp = np.logspace(-9, -4, 100)
    # TODO: move these to r_grid_to_merge
    # add Aeronet and Drewnick radii for normalization
    # dp = np.append(dp, 2*AERONET_NORMALIZATION_RADUIS * 10 ** -6)
    # dp = np.append(dp, 2*DREWNICK_NORMALIZATION_RADUIS * 10 ** -6)
    if r_grid_to_merge is not None:
        dp = np.append(dp, r_grid_to_merge*2)
    dp.sort()

    radii = dp / 2
    dNdlogp_list = sample_WRF_MADE_size_distributions(dp, sgs, dgs, m0s, m3s)

    vo = {}
    vo['data'] = np.array(dNdlogp_list)  # this SD will be total
    vo['radii'] = radii * 10 ** 6  # um

    time_key = 'time'
    if time_key not in nc.variables.keys():
        time_key = 'XTIME'
    vo['time'] = convert_time_data_impl(nc.variables[time_key][:], nc.variables[time_key].units)

    if isinstance(vo['time'], np.ma.masked_array):
        vo['time'] = vo['time'].filled()  # prevent time being masked array

    # Aux stuff
    if sum_up_modes:
        vo['data'] = np.sum(vo['data'], axis=0)

    if column:  # integrate vertically
        z_stag = nc['PH'][:] + nc['PHB'][:] / 9.8
        dz = np.diff(z_stag, axis=1)  # m
        z_dim = vo['data'].shape.index(nc.dimensions['bottom_top'].size)  # deduce z_dim index
        vo['data'] = np.sum(vo['data'] * dz[..., np.newaxis], axis=z_dim)
        # TODO: do not change units here
        vo['data'] *= 10 ** -12  # particles * um**3 / um**2  # convert units: WRF [part * um^3 / m^3 * m] to Aeronet [um^3/um^2]

    return vo
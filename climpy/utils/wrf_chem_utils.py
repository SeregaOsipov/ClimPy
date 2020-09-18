import numpy as np
import scipy as sp
import scipy.special
import functools

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

CHEM_106_AEROSOLS_KEYS = 'so4aj,so4ai,nh4aj,nh4ai,no3aj,no3ai,naaj,naai,claj,clai,orgaro1j,orgaro1i,orgaro2j,orgaro2i,orgalk1j,orgalk1i,orgole1j,orgole1i,orgba1j,orgba1i,orgba2j,orgba2i,orgba3j,orgba3i,orgba4j,orgba4i,orgpaj,orgpai,ecj,eci,p25j,p25i,antha,seas,soila'.split(',')

# !  initial mean diameter for nuclei mode, accumulation and coarse modes [ m ]
#       PARAMETER (dginin=0.01E-6)
#       PARAMETER (dginia=0.07E-6)
#       PARAMETER (dginic=1.0E-6)
MADE_MODES_DGs = [0.01E-6, 0.07E-6, 1.0E-6]
# !  initial sigma-G for nuclei, acc, and coarse modes
#       PARAMETER (sginin=1.70)
#       PARAMETER (sginia=2.00)
#       PARAMETER (sginic=2.5)
MADE_MODES_SIGMA = [1.7, 2.0, 2.5]


def get_WRF_MADE_initial_params():
    '''
    This routine mimics the emitted size distribution of the dust in WRF Chem chem_opt=106

    NOTE: the WRF-Chem source code multiplies coarse mode fraction 0.93 by 1.02 (corfrac=corfrac*rscale2) to acount for >10 um particles
    NOTE: They also use different densities for different modes, which is ignored here

    :return: the default parameters of the initial size distributions given by MADE in WRF-Chem
    '''

    dgs = MADE_MODES_DGs
    sgs = MADE_MODES_SIGMA

    moment3s = [0, 7 / 100, 93 / 100]  # fraction of the dust mass distributed by modes
    moment0s = []  # derived from m3s

    for sg, dg, m3 in zip(sgs, dgs, moment3s):
        es36 = np.exp(0.125 * np.log(sg) ** 2) ** 36
        # chem(i,k,j,p_ac0) = m3acc/((dginia**3)*esa36)
        m0 = m3 / ((dg**3) * es36)
        moment0s.append(m0)

    return sgs, dgs, moment0s, moment3s


def get_WRF_MADE_modpar(moment0_list, moment3_list, sg_list):
    """
    see ACKERMANN et al., 1998 MODAL AEROSOL DYNAMICS MODEL FOR EUROPE: DEVELOPMENT AND FIRST APPLICATIONS
    https://www.sciencedirect.com/science/article/abs/pii/S1352231098000065

    code was ported from WRF Chem module_aerosols_sorgam.f (and data)

    :param moment0_list = (nu0, ac0, cor0)  # 0 moments
    :param moment3_list = (nu3, ac3, cor3)  # 3rd moments
    :param sg_list = (1.7, 2.0, 2.5)  # sginin, sginia, sginic # initial sigma-G for nuclei, acc, coarse modes

    :return: median diameters computed as:
    dgacc(lcell) = max(dgmin,(cblk(lcell,vac3)/(cblk(lcell,vac0)*esa36))**one3)
    """

    # TODO add cw_phase logic, see the source code
    dg_list = ()
    dgmin = 1.0E-09
    for sgini, moment3, moment0 in zip(sg_list, moment3_list, moment0_list):
        es36 = np.exp(0.125 * np.log(sgini) ** 2) ** 36
        dg = (moment3 / (moment0 * es36)) ** (1 / 3)
        dg[dg < dgmin] = dgmin
        dg_list += (dg,)

    # sginin = 1.70
    # esn36 = np.exp(0.125*np.log(sginin) ** 2) ** 36
    # dgacc = (ac3/ac0*esn36)**(1/3)
    # dgmin = 1.0E-09
    # dgacc[dgacc < dgmin] = dgmin

    return dg_list


def sample_WRF_MADE_size_distributions(dp, sg_list, dg_list, moment3_list, moment0_list):
    """

    :param dp:
    :param sg_list:
    :param dg_list:
    :param moment3_list:
    :param moment0_list:
    :return: the dNdlog(p) size distribution for each mode and scaled to total number of particles
    """
    dNdlogp_list = ()
    for sg, dg, moment3, moment0 in zip(sg_list, dg_list, moment3_list, moment0_list):

        # if sg/dg... are numbers, then convert them to arrays
        if isinstance(sg, float):
            sg = np.array(sg)
            dg = np.array(dg)
            moment3 = np.array(moment3)
            moment0 = np.array(moment0)

        # THIS one is faster then sp.stats.lognorm
        dNdlogp = 1/((2*np.pi)**(1/2) * np.log(sg[..., np.newaxis])) * np.exp(-1/2 * (np.log(dp)-np.log(dg[..., np.newaxis]))**2 / np.log(sg[..., np.newaxis])**2)
        # lognorm_dist = sp.stats.lognorm(s=np.log(sg[..., np.newaxis]), loc=0, scale=dg[..., np.newaxis])
        # dNdlogp = dp * lognorm_dist.pdf(dp)
        dNdlogp *= moment0[..., np.newaxis]  # this is the n(logdp) from ACKERMANN et al., equation 1
        dNdlogp_list += (dNdlogp,)

    return dNdlogp_list


def compute_MADE_bounded_distribution_factors(d_min, d_max, sg_list, dg_list, moment3_list, moment0_list):
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
    for sg, dg, moment3, moment0 in zip(sg_list, dg_list, moment3_list, moment0_list):
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


def get_chemistry_package_definition(chem_opt):
    """
    Copy-pasted from registry.chem
    :param chem_opt:
    :return: list of keys to access netcdf variables
    """
    # chem_opt specific output, was copied from the registry.chem
    if chem_opt == 105:  # package racmsorg_kpp
        package_vars = 'so2,sulf,no2,no,o3,hno3,h2o2,ald,hcho,op1,op2,paa,ora1,ora2,nh3,n2o5,no3,pan,hc3,hc5,hc8,eth,co,ete,olt,oli,tol,xyl,aco3,tpan,hono,hno4,ket,gly,mgly,dcb,onit,csl,iso,co2,ch4,udd,hket,api,lim,dien,macr,ho,ho2,so4aj,so4ai,nh4aj,nh4ai,no3aj,no3ai,naaj,naai,claj,clai,orgaro1j,orgaro1i,orgaro2j,orgaro2i,orgalk1j,orgalk1i,orgole1j,orgole1i,orgba1j,orgba1i,orgba2j,orgba2i,orgba3j,orgba3i,orgba4j,orgba4i,orgpaj,orgpai,ecj,eci,p25j,p25i,antha,seas,soila,nu0,ac0,corn'.split(',')
    if chem_opt == 106:  # package   radm2sorg_kpp
        package_vars = 'so2,sulf,no2,no,o3,hno3,h2o2,ald,hcho,op1,op2,paa,ora1,ora2,nh3,n2o5,no3,pan,hc3,hc5,hc8,eth,co,ol2,olt,oli,tol,xyl,aco3,tpan,hono,hno4,ket,gly,mgly,dcb,onit,csl,iso,co2,ch4,ho,ho2,so4aj,so4ai,nh4aj,nh4ai,no3aj,no3ai,naaj,naai,claj,clai,orgaro1j,orgaro1i,orgaro2j,orgaro2i,orgalk1j,orgalk1i,orgole1j,orgole1i,orgba1j,orgba1i,orgba2j,orgba2i,orgba3j,orgba3i,orgba4j,orgba4i,orgpaj,orgpai,ecj,eci,p25j,p25i,antha,seas,soila,nu0,ac0,corn'
    if chem_opt == 301:  # package   gocartracm_kpp
        package_vars = 'so2,sulf,no2,no,o3,hno3,h2o2,ald,hcho,op1,op2,paa,ora1,ora2,nh3,n2o5,no3,pan,hc3,hc5,hc8,eth,co,ete,olt,oli,tol,xyl,aco3,tpan,hono,hno4,ket,gly,mgly,dcb,onit,csl,iso,co2,ch4,udd,hket,api,lim,dien,macr,ho,ho2,dms,msa,p25,bc1,bc2,oc1,oc2,dust_1,dust_2,dust_3,dust_4,dust_5,seas_1,seas_2,seas_3,seas_4,p10'

    return package_vars.split(',')


def combine_aerosol_keys_by_size_modes(aerosol_keys):
    '''
    # in the future this can be implemented for other schemes as well
    :param aerosol_keys: only CHEM_106_AEROSOLS_KEYS
    :return: list of keys for each size mode
    '''

    # output lists
    nucleation_keys = []
    accumulation_keys = []
    coarse_keys = []
    for key in aerosol_keys:
        if key[-1] == 'i':
            nucleation_keys.append(key)
        elif key[-1] == 'j':
            accumulation_keys.append(key)
        else:  # antha, seas, soila. Leave them as is
            coarse_keys.append(key)

    keys_by_modes = [nucleation_keys, accumulation_keys, coarse_keys]
    mode_labels = ['Nucleation', 'Accumulation', 'Coarse']
    return keys_by_modes, mode_labels


# Next is 3 decorator and routine to get the stack of all aerosols
# This was designed to work with MADE aerosol scheme


def vstack_and_sort_aerosols(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):

        diags, aerosols_keys = func(*args, **kwargs)

        # stack all species together
        diags_vstack = np.stack(diags, axis=0)

        return diags_vstack, aerosols_keys

        # sort them, dims are species, time
        ind = np.argsort(np.nansum(diags_vstack, axis=1))
        # Reverse the sorted array
        ind = ind[::-1]

        ind_3d = np.repeat(ind[:, np.newaxis, :], diags_vstack.shape[1], axis=1)
        np.take_along_axis(diags_vstack, ind_3d, axis=1).shape

        return np.take_along_axis(diags_vstack, ind_3d, axis=1), np.array(aerosols_keys)[ind[:, 0]]

    return wrapper_decorator


def combine_aerosol_modes(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        combine_modes = None
        if 'combine_modes' in kwargs:
            combine_modes = kwargs.pop('combine_modes')

        diags, aerosols_keys = func(*args, **kwargs)

        diags_pp = diags
        aerosols_keys_pp = aerosols_keys

        if combine_modes:  # merge two modes into one, for example, so4ai+so4aj = so4ai+j
            diags_pp = []
            aerosols_keys_pp = []
            for key, diag in zip(aerosols_keys, diags):
                # NOTE: this logic wil break, if the var names change
                if key[-1] == 'i':
                    i_index = aerosols_keys.index(key[:-1] + "i")
                    j_index = aerosols_keys.index(key[:-1] + "j")
                    diags_pp.append(diags[i_index] + diags[j_index])
                    aerosols_keys_pp.append(key[:-1]+' (i+j)')
                elif key[-1] == 'j':
                    continue  # it was proccessed by the previous case
                else:  # antha, seas, soila. Leave them as is
                    diags_pp.append(diag)
                    aerosols_keys_pp.append(key)
        return diags_pp, aerosols_keys_pp

    return wrapper_decorator


def combine_aerosol_types(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        combine_subtypes = None
        if 'combine_subtypes' in kwargs:
            combine_subtypes = kwargs.pop('combine_subtypes')

        diags, aerosols_keys = func(*args, **kwargs)

        diags_pp = diags
        aerosols_keys_pp = aerosols_keys

        if combine_subtypes:  # merge organics org..1 + org..2 into org (1+2)
            diags_pp = []
            aerosols_keys_pp = []
            organics = []
            for key, diag in zip(aerosols_keys, diags):
                # simply group all organics into one bucket
                # indices = [i for i, s in enumerate(aerosols_keys) if 'org' in s]
                if key[:3] == 'org':
                    organics.append(diag)
                else:  # everything else leave as is
                    diags_pp.append(diag)
                    aerosols_keys_pp.append(key)
            if len(organics) > 0:
                # sum up organics
                diags_pp.append(sum(organics))
                aerosols_keys_pp.append('org')
        return diags_pp, aerosols_keys_pp

    return wrapper_decorator


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
        sg_list, dg_list, moment0_list, moment3_list = aqaba.get_wrf_sd_params(nc)
        d_min = pm_sizes[0]
        d_max = pm_sizes[1]
        N_factors, V_factors = wrf_chem.compute_MADE_bounded_distribution_factors(d_min, d_max, sg_list, dg_list, moment3_list, moment0_list)

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

        diags.append(np.squeeze(diag))  # [ug m**-3]

    return diags, aerosols_keys


def rank_aerosols_contribution_to_the_mode(nc, aerosols_keys):
    '''
    This routine compute the contribution of the aerosol type to the size mode
    :param nc:
    :param aerosols_keys:
    :return:
    '''
    diags_vstack, keys = get_aerosols_stack(nc, aerosols_keys, combine_subtypes=True)

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


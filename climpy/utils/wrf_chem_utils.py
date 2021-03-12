import pandas as pd
import numpy as np
import functools
from climpy.utils.diag_decorators import pandas_time_interval_selection
from climpy.utils.netcdf_utils import convert_time_data_impl

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'


"""
For chem_opt=100:
    p_so4aj=75, num_chem=118
    which means that alt applied to all aerosols and water, except m0s and m3s
    
    NOTE: also that m3s are wet in WRF, although water is diagnostic and not a tracer
"""

# Don't forget to drop nu0,ac0,corn. Cast to tuple to keep them unchangable
CHEM_100_AEROSOLS_KEYS = tuple('so4aj,so4ai,nh4aj,nh4ai,no3aj,no3ai,naaj,naai,claj,clai,asoa1j,asoa1i,asoa2j,asoa2i,asoa3j,asoa3i,asoa4j,asoa4i,bsoa1j,bsoa1i,bsoa2j,bsoa2i,bsoa3j,bsoa3i,bsoa4j,bsoa4i,orgpaj,orgpai,ecj,eci,caaj,caai,kaj,kai,mgaj,mgai,p25j,p25i,antha,seas,soila'.split(','))
CHEM_106_AEROSOLS_KEYS = tuple('so4aj,so4ai,nh4aj,nh4ai,no3aj,no3ai,naaj,naai,claj,clai,orgaro1j,orgaro1i,orgaro2j,orgaro2i,orgalk1j,orgalk1i,orgole1j,orgole1i,orgba1j,orgba1i,orgba2j,orgba2i,orgba3j,orgba3i,orgba4j,orgba4i,orgpaj,orgpai,ecj,eci,p25j,p25i,antha,seas,soila'.split(','))
CHEM_108_AEROSOLS_KEYS = tuple('so4aj,so4ai,nh4aj,nh4ai,no3aj,no3ai,naaj,naai,claj,clai,asoa1j,asoa1i,asoa2j,asoa2i,asoa3j,asoa3i,asoa4j,asoa4i,bsoa1j,bsoa1i,bsoa2j,bsoa2i,bsoa3j,bsoa3i,bsoa4j,bsoa4i,orgpaj,orgpai,ecj,eci,p25j,p25i,antha,seas,soila'.split(','))


def get_molecule_key_from_aerosol_key(aerosol_key):
    '''
    Strip i or j modes and letter a
    '''

    molecule_key = None

    exceptions = ['antha', 'seas', 'soila']
    if aerosol_key in exceptions:
        if aerosol_key == 'seas':
            molecule_key = aerosol_key
        else:
            molecule_key = aerosol_key[:-1]
        return molecule_key

    only_mode_exceptions = ['bsoa', 'asoa', 'ec', 'p25']
    for exception in only_mode_exceptions:
        if exception in aerosol_key:
            molecule_key = aerosol_key[:-1]
            return molecule_key

    # everything else, regular names
    molecule_key = aerosol_key[:-2]

    return molecule_key.lower()


def get_chemistry_package_definition(chem_opt):
    """
    Copy-pasted from registry.chem
    :param chem_opt:
    :return: list of keys to access netcdf variables
    """
    # chem_opt specific output, was copied from the registry.chem
    if chem_opt == 100:  # package racm_soa_vbs_het_kpp
        package_vars = 'so2,sulf,no2,no,o3,hno3,h2o2,ald,hcho,op1,op2,paa,ora1,ora2,nh3,n2o5,no3,pan,hc3,hc5,hc8,eth,co,ete,olt,oli,tol,xyl,aco3,tpan,hono,hno4,ket,gly,mgly,dcb,onit,csl,iso,co2,ch4,udd,hket,api,lim,dien,macr,hace,ishp,ison,mahp,mpan,nald,sesq,mbo,cvasoa1,cvasoa2,cvasoa3,cvasoa4,cvbsoa1,cvbsoa2,cvbsoa3,cvbsoa4,hcl,clno2,cl2,fmcl,cl,clo,hocl,ch3cl,ho,ho2,so4aj,so4ai,nh4aj,nh4ai,no3aj,no3ai,naaj,naai,claj,clai,asoa1j,asoa1i,asoa2j,asoa2i,asoa3j,asoa3i,asoa4j,asoa4i,bsoa1j,bsoa1i,bsoa2j,bsoa2i,bsoa3j,bsoa3i,bsoa4j,bsoa4i,orgpaj,orgpai,ecj,eci,caaj,caai,kaj,kai,mgaj,mgai,p25j,p25i,antha,seas,soila,nu0,ac0,corn'
    if chem_opt == 105:  # package racmsorg_kpp
        package_vars = 'so2,sulf,no2,no,o3,hno3,h2o2,ald,hcho,op1,op2,paa,ora1,ora2,nh3,n2o5,no3,pan,hc3,hc5,hc8,eth,co,ete,olt,oli,tol,xyl,aco3,tpan,hono,hno4,ket,gly,mgly,dcb,onit,csl,iso,co2,ch4,udd,hket,api,lim,dien,macr,ho,ho2,so4aj,so4ai,nh4aj,nh4ai,no3aj,no3ai,naaj,naai,claj,clai,orgaro1j,orgaro1i,orgaro2j,orgaro2i,orgalk1j,orgalk1i,orgole1j,orgole1i,orgba1j,orgba1i,orgba2j,orgba2i,orgba3j,orgba3i,orgba4j,orgba4i,orgpaj,orgpai,ecj,eci,p25j,p25i,antha,seas,soila,nu0,ac0,corn'
    if chem_opt == 106:  # package   radm2sorg_kpp
        package_vars = 'so2,sulf,no2,no,o3,hno3,h2o2,ald,hcho,op1,op2,paa,ora1,ora2,nh3,n2o5,no3,pan,hc3,hc5,hc8,eth,co,ol2,olt,oli,tol,xyl,aco3,tpan,hono,hno4,ket,gly,mgly,dcb,onit,csl,iso,co2,ch4,ho,ho2,so4aj,so4ai,nh4aj,nh4ai,no3aj,no3ai,naaj,naai,claj,clai,orgaro1j,orgaro1i,orgaro2j,orgaro2i,orgalk1j,orgalk1i,orgole1j,orgole1i,orgba1j,orgba1i,orgba2j,orgba2i,orgba3j,orgba3i,orgba4j,orgba4i,orgpaj,orgpai,ecj,eci,p25j,p25i,antha,seas,soila,nu0,ac0,corn'
    if chem_opt == 108:  # package   racm_soa_vbs_kpp
        package_vars = 'so2,sulf,no2,no,o3,hno3,h2o2,ald,hcho,op1,op2,paa,ora1,ora2,nh3,n2o5,no3,pan,hc3,hc5,hc8,eth,co,ete,olt,oli,tol,xyl,aco3,tpan,hono,hno4,ket,gly,mgly,dcb,onit,csl,iso,co2,ch4,udd,hket,api,lim,dien,macr,hace,ishp,ison,mahp,mpan,nald,sesq,mbo,cvasoa1,cvasoa2,cvasoa3,cvasoa4,cvbsoa1,cvbsoa2,cvbsoa3,cvbsoa4,ho,ho2,so4aj,so4ai,nh4aj,nh4ai,no3aj,no3ai,naaj,naai,claj,clai,asoa1j,asoa1i,asoa2j,asoa2i,asoa3j,asoa3i,asoa4j,asoa4i,bsoa1j,bsoa1i,bsoa2j,bsoa2i,bsoa3j,bsoa3i,bsoa4j,bsoa4i,orgpaj,orgpai,ecj,eci,p25j,p25i,antha,seas,soila,nu0,ac0,corn'
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
        combine_organics = None
        if 'combine_organics' in kwargs:
            combine_organics = kwargs.pop('combine_organics')

        combine_sea_salt = None
        if 'combine_sea_salt' in kwargs:
            combine_sea_salt = kwargs.pop('combine_sea_salt')

        diags, aerosols_keys = func(*args, **kwargs)

        diags_pp = diags
        aerosols_keys_pp = aerosols_keys

        if combine_organics or combine_sea_salt:  # merge organics org..1 + org..2 into org (1+2)
            diags_pp = []
            aerosols_keys_pp = []
            organics = []
            minerals = []
            for key, diag in zip(aerosols_keys, diags):
                # simply group all organics into one bucket
                # indices = [i for i, s in enumerate(aerosols_keys) if 'org' in s]
                if combine_organics and (key[:3] == 'org' or key[:4] == 'asoa' or key[:4] == 'bsoa'):
                    organics.append(diag)
                elif combine_sea_salt and (key == 'seas' or key[:3] == 'caa' or key[:2] == 'ka' or key[:3] == 'mga' or key[:3] == 'naa' or key[:3] == 'cla'):  # Ca, K, Mg
                    minerals.append(diag)
                else:  # everything else leave as is
                    diags_pp.append(diag)
                    aerosols_keys_pp.append(key)
            if len(organics) > 0:  # sum up organics
                diags_pp.append(sum(organics))
                aerosols_keys_pp.append('org (i+j)')
            if len(minerals) > 0:
                diags_pp.append(sum(minerals))
                aerosols_keys_pp.append('seas*')  # caa+ka+mga+naa+cla
        return diags_pp, aerosols_keys_pp

    return wrapper_decorator


def to_stp(func):
    '''
    Normalize to standard pressure and temperature
    p_stp = 1013.25 hPa, t_stp = 20 C

    C_stp = p_stp/p * t/t_stp * C
    '''
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):

        convert_to_stp = None
        if 'convert_to_stp' in kwargs:
            convert_to_stp = kwargs.pop('convert_to_stp')

        diags, aerosols_keys = func(*args, **kwargs)

        if convert_to_stp is not None:
            print('Normalizing concentrations to STP')

            nc = args[0]
            # TODO: check the height of the instruments
            t = nc.variables['T2'][:]
            p = nc.variables['PSFC'][:]*10**-2

            p_stp = 1013.25
            t_stp = 273+20
            stp_correction_factor = p_stp/p * t/t_stp
            print('stp correction min is {}, max is {}'.format(np.min(stp_correction_factor), np.max(stp_correction_factor)))

            for diag in diags:
                diag *= stp_correction_factor[:, np.newaxis]

        return diags, aerosols_keys

    return wrapper_decorator


@to_stp
# @ppmv_to_ugm3
def get_gas_phase_organics_stack(nc, gas_keys):
    """
    Computes the aerosols PM mass concentration

    Examples:
    To compute PM for [d>250nm and < 10um] set pm_sizes = [0.25 * 10 ** -6, 10 * 10 ** -6]  # m

    :param nc:
    :param gas_keys:
    :param pm_sizes: [d_min, d_max] in meters,
    :return:
    """

    diags = []
    alt = nc.variables['ALT'][:]
    for key in gas_keys:
        # gas units are ppmv, alt is [m3 kg-1]
        # convert to volume concentration [ug m^-3]
        # vmr -> mmr -> volume concentration
        diag = nc.variables[key][:] / alt

        diags.append(np.squeeze(diag))  # [ug m**-3]

    return diags, gas_keys


def get_aerosols_keys(chem_opt, wet=True, non_refractory=False):
    """
    non_refractory aersols are those that can be measured by AMS (aerosols mass spectrometer),
    i.e. those, that evaporate (when impacted onto a 600 C hot plate)
    """

    keys = None
    if chem_opt == 100:
        keys = CHEM_100_AEROSOLS_KEYS
    elif chem_opt == 106:
        keys = CHEM_106_AEROSOLS_KEYS
    elif chem_opt == 108:
        keys = CHEM_108_AEROSOLS_KEYS

    if wet:
        keys += ('H2OAI', 'H2OAJ')

    if non_refractory:  # this will probably only work for MADE scheme
        keys = list(keys)
        # not sure if antha is refractory
        not_in_AMS_keys = ['soila', 'seas', 'p25j', 'p25i', 'antha'] + ['naai', 'nnaaj', 'caai', 'caaj', 'kai', 'kaj', 'mgai', 'mgaj']  # extra keys from 100 case
        not_in_AMS_keys += ['clai', 'claj']  # exclude Chlorine as well, because AMS measured only 1% of it in PM1
        for key in not_in_AMS_keys:
            if key in keys:
                keys.remove(key)

    return keys


def get_organic_aerosols_keys(chem_opt):
    """
    Return the anthropogenic and biogenic keys
    """

    asoa_keys = None
    bsoa_keys = None
    if chem_opt == 106:
        asoa_keys = ('orgaro1i', 'orgaro1j', 'orgaro2i', 'orgaro2j', 'orgalk1i', 'orgalk1j', 'orgole1i', 'orgole1j')  # SOA Anth
        bsoa_keys = ('orgba4i', 'orgba4j', 'orgba3i', 'orgba3j', 'orgba2i', 'orgba2j', 'orgba1i', 'orgba1j')  # SOA Biog
    elif chem_opt == 108 or chem_opt == 100:
        asoa_keys = 'asoa1j,asoa1i,asoa2j,asoa2i,asoa3j,asoa3i,asoa4j,asoa4i'.split(',')  # SOA Anth
        bsoa_keys = 'bsoa1j,bsoa1i,bsoa2j,bsoa2i,bsoa3j,bsoa3i,bsoa4j,bsoa4i'.split(',')  # SOA Biog
    else:
        print('PP: this chem_opt {} is not implemented, dont know how to combine organics')

    return asoa_keys, bsoa_keys


def get_molecular_weight(key):
    wrf_to_iso_map = {}
    wrf_to_iso_map['HC5'] = 'C8H10'  # Xylene
    wrf_to_iso_map['XYL'] = 'C8H10'  # Xylene
    wrf_to_iso_map['XYL'] = 'C8H10'  # Xylene
    wrf_to_iso_map['XYL'] = 'C8H10'  # Xylene

    molecular_weights = {}
    molecular_weights['HC5'] = 'C8H10'  # Xylene
    molecular_weights['XYL'] = 'C8H10'  # Xylene
    molecular_weights['XYL'] = 'C8H10'  # Xylene
    molecular_weights['XYL'] = 'C8H10'  # Xylene

    #Formula(key).
    return


# 2D data from pp-ed WRF output to match Aeronet analysis


@pandas_time_interval_selection
def get_wrf_aod_at_aeronet_station(nc, var_key):
    '''
    Prepare the column AOD at aeronet location from pp-ed WRF file
    :param nc:
    :param var_key:
    :return:
    '''
    vo = {}
    vo['data'] = nc.variables[var_key][:]

    # derive column AOD
    vo['data'] = np.sum(vo['data'], axis=1)
    vo['data'] = np.squeeze(vo['data'])

    time_key = 'time'
    if time_key not in nc.variables.keys():
        time_key = 'XTIME'
    vo['time'] = convert_time_data_impl(nc[time_key][:], nc[time_key].units)

    # convert to pandas, since data is 2D
    df = pd.DataFrame({var_key: vo['data']}, pd.DatetimeIndex(vo['time']))

    return df



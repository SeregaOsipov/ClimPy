import functools
import netCDF4
import numpy as np
from climpy.utils.file_path_utils import get_root_storage_path_on_hpc
from climpy.utils.wrf_chem_made_utils import get_aerosols_stack
from climpy.utils.wrf_chem_utils import get_aerosols_keys, get_molecule_key_from_aerosol_key

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'


def interpolate_ri_in_wavelength(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        wavelengths = None
        if 'wavelengths' in kwargs:
            wavelengths = kwargs.pop('wavelengths')

        vo = func(*args, **kwargs)

        if wavelengths is not None:
            # interpolate RI onto internal wavelength grid
            ri = np.interp(wavelengths, vo['wl'], vo['ri'])  # ri = real+1j*imag
            vo['ri'] = ri
            vo['wl'] = wavelengths

        return vo
    return wrapper_decorator


def correct_ri_sign_convention(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        sign_convention = 'negative'
        if 'sign_convention' in kwargs:
            sign_convention = kwargs.pop('sign_convention')

        vo = func(*args, **kwargs)

        if sign_convention is not None and sign_convention == 'negative':
            if vo['ri'] is np.array:
                ri = [ri.real - 1j * ri.imag for ri in vo['ri']]
            else:
                ri = vo['ri'].real - 1j * vo['ri'].imag
            vo['ri'] = ri
        return vo
    return wrapper_decorator


@interpolate_ri_in_wavelength
def get_Williams_Palmer_refractive_index():
    '''
    :return: Refractive index from Palmer and Williams(1975) of 75 % sulfate aerosol
    wavelength is in microns
    '''

    nc_fp = get_root_storage_path_on_hpc() + '/Data/Harvard/HITRAN/HITRAN2012/Aerosols/netcdf/palmer_williams_h2so4.nc'
    nc = netCDF4.Dataset(nc_fp)
    wl = nc.variables['wavelength'][:]
    ri_real = nc.variables['rn'][3, :]  # index 4 is 75% solution
    ri_imag = nc.variables['ri'][3, :]

    ri_vo = {}
    ri_vo['ri'] = ri_real + 1j * ri_imag
    ri_vo['ri'] = ri_vo['ri'][::-1]
    ri_vo['wl'] = wl[::-1]
    return ri_vo


@correct_ri_sign_convention
@interpolate_ri_in_wavelength
def get_dust_ri():
    '''
    SW: https://data.eurochamp.org/data-access/optical-properties/
    LW: https://acp.copernicus.org/articles/17/1901/2017/
    :return:
    '''

    wl = np.array([0.52, 0.95, 10])
    ri_real = np.array([1.54, 1.54, 1.602])
    ri_imag = np.array([15*10**-4, 6*10**-4, 0.28])

    ri_vo = {}
    ri_vo['ri'] = ri_real + 1j * ri_imag
    ri_vo['wl'] = wl
    return ri_vo


@correct_ri_sign_convention
def get_spectral_refractive_index(chem_key, wavelengths):
    '''
    Simplified version to get RI
    It mimics WRF-Chem and returns spectrally gray RIs

    wavelengths: in um
    '''
    # Simplified list of species for visible, page 13 in https://www.acom.ucar.edu/webt/fundamentals/2018/Lecture11_Barth.pdf
    # Also see module_optical_averaging.F, line 1884 for a list of WRF RI
    # all the values were taken from the module_data_rrtmgaeropt.F
    # TODO: WRF miimicing is incomplete
    wl = 0.5  # um
    ri = None

    if chem_key in ['bc', 'ec']:
        ri = 1.95 + 1j * 0.79  # 1.85 + 1j * 0.71
    elif chem_key in ['oc', 'orgp'] or chem_key[1:4] == 'soa':  # special case for asoa4 and such
        ri = 1.45 + 1j * 0
    elif chem_key in ['p25', 'anth']:  # assume the same as for CaSO4, borrowed from WRF
        ri = 1.56 + 1j * 6*10**-3
    elif chem_key == 'so4':
        ri = 1.52 + 1j * 10**-9
    elif chem_key == 'nh4':  # assume the same as for NH4NO3
        ri = 1.5 + 1j * 0
    elif chem_key == 'no3':  # assume the same as for NH4NO3  # TODO find individual refractive indecies
        ri = 1.5 + 1j * 0
    elif chem_key == 'na':  # assume the same as for NaCl
        ri = 1.51 + 1j * 0.866*10**-6
    elif chem_key == 'cl':  # assume the same as for NaCl
        ri = 1.51 + 1j * 0.866*10**-6
    elif chem_key == 'ca':  # assume the same as for CaSO4, borrowed from WRF
        ri = 1.56 + 1j * 6*10**-3
    elif chem_key == 'k':  #
        ri = 1.48814 + 1j * 0
    elif chem_key == 'mg':  #
        ri = 1.73 + 1j * 0
    # elif chem_key == 'dust':
    #     ri = 1.55 + 1j * 3*10 ** -3
    elif chem_key == 'seas':  # TODO: update
        ri = 1.5 + 1j * 7.019*10**-8  # module_data_rrtmgaeropt.F, line 63
    elif chem_key == 'soil':
        ri = 1.55 + 1j * 3*10**-3  # module_data_rrtmgaeropt.F, line 32
    elif chem_key == 'h2o':  # TODO: update
        ri = 1.35 + 1j * 1.52*10**-8
    else:
        raise Exception('key {} is not found'.format(chem_key))

    wls = wavelengths
    if type(wavelengths) is list:
        wls = np.array(wavelengths)

    ri_vo = {}  # return spectral RI
    ri_vo['ri'] = np.ones(wls.shape) * ri  # ri
    ri_vo['wl'] = wls  # wl
    ri_vo['key'] = chem_key

    return ri_vo


def mix_refractive_index(nc, chem_opt, wavelengths=None):
    '''
    WRF specific implementation
    Compute the volume-weighted refractive for the entire column

    wavelengths: in um
    '''

    if wavelengths is None:
        wavelengths = np.array([0.3, 0.4, 0.6, 0.999])  # default WRF wavelengths in SW

    aerosols_keys_wet = get_aerosols_keys(chem_opt, wet=True)
    aerosols_stack, dummy = get_aerosols_stack(nc, aerosols_keys_wet)  # , pm_sizes=[d_min, d_max], combine_modes=True, combine_organics=True, combine_sea_salt=True, convert_to_stp=True)

    # the weight will be aerosols mass, integrate vertically
    z_stag = nc['PH'][:] + nc['PHB'][:] / 9.8
    dz = np.diff(z_stag, axis=1)  # m
    z_dim = aerosols_stack.shape.index(nc.dimensions['bottom_top'].size)  # deduce z_dim index
    weights = np.sum(aerosols_stack*dz, axis=z_dim)  # ug / m^2

    ri_vos = []
    ris = []
    for aerosol_key in aerosols_keys_wet:
        key = get_molecule_key_from_aerosol_key(aerosol_key)
        ri = get_spectral_refractive_index(key, wavelengths)
        ri_vos.append(ri)
        ris.append(ri['ri'])

    # reshape RIs to match the array shapes, dims: chem, time, wl
    ris = np.tile(np.array(ris), weights.shape[1:] + (1, 1))
    ris = np.moveaxis(ris, -2, 0)
    weights = np.tile(weights[..., np.newaxis], (wavelengths.shape[0],))

    ri_volume_weighted = np.average(ris, weights=weights, axis=0)  # apply weights for averaging
    ri_vo = {'ri': ri_volume_weighted, 'wl': wavelengths}

    return ri_vo
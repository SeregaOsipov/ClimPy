import functools
import netCDF4
import numpy as np
import xarray as xr
from climpy.utils.file_path_utils import get_root_storage_path_on_hpc
from climpy.utils.wrf_chem_made_utils import get_aerosols_pm_stack
from climpy.utils.wrf_chem_utils import get_aerosols_keys, get_molecule_key_from_aerosol_key

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'


def interpolate_ri_in_wavelength(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        wavelengths = None
        if 'wavelengths' in kwargs:
            wavelengths = kwargs.pop('wavelengths')

        ds = func(*args, **kwargs)

        if wavelengths is not None:
            ds = ds.interp(wavelength=wavelengths, kwargs={"fill_value": (ds.isel(wavelength=0), ds.isel(wavelength=-1))},)

        return ds
    return wrapper_decorator


def correct_ri_sign_convention(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        sign_convention = 'negative'
        if 'sign_convention' in kwargs:
            sign_convention = kwargs.pop('sign_convention')

        ds = func(*args, **kwargs)

        if sign_convention is not None and sign_convention == 'negative':
            if isinstance(ds, xr.DataArray):
                ds = ds.real - 1j * ds.imag
            else:  # TODO: replace everything with xarray
                if ds['ri'] is np.array:
                    ri = [ri.real - 1j * ri.imag for ri in ds['ri']]
                else:
                    ri = ds['ri'].real - 1j * ds['ri'].imag
                ds['ri'] = ri
        return ds
    return wrapper_decorator


#@interpolate_ri_in_wavelength
def get_Williams_Palmer_refractive_index():
    '''
    :return: Refractive index from Palmer and Williams(1975) of 75 % sulfate aerosol
    wavelength is in microns

    url: https://hitran.org/data/Aerosols/Aerosols-2020/
    '''

    # nc_fp = get_root_storage_path_on_hpc() + '/Data/Harvard/HITRAN/HITRAN2012/Aerosols/netcdf/palmer_williams_h2so4.nc'
    nc_fp = get_root_storage_path_on_hpc() + '/Data/HITRAN/Aerosols/Aerosols-2020/hitran_ri/netcdf/palmer_williams_h2so4.nc'
    nc = netCDF4.Dataset(nc_fp)

    ri_real = nc.variables['rn'][3, :]  # index 4 is 75% solution
    ri_imag = nc.variables['ri'][3, :]

    ds = xr.Dataset(
        data_vars=dict(
            ri=(["wavelength", ], ri_real + 1j * ri_imag),
        ),
        coords=dict(
            wavelength=("wavelength", nc.variables['wavelength'][:]),
        ),
        attrs=dict(description="Palmer and Williams 75% H2SO4 complex refractive index."),
    )
    ds = ds.sortby('wavelength')
    return ds


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

    ri_ds = xr.DataArray(data=ri_real + 1j * ri_imag, dims='wavelength', coords=dict(wavelength=wl), name='ri')
    return ri_ds


def get_dust_WRF_Stenchikov_ri():
    '''
    The WRF and Stenchikov specific version of dust RI
    Suleiman provided the number. These are not default!
    :return:
    '''

    wn_stag_lw = np.array([10, 350, 500, 630, 700, 820, 980, 1080, 1180, 1390, 1480, 1800, 2080, 2250, 2380, 2600, 3250])  # this is non-default
    wn_stag_sw = np.array([820, 2600, 3250, 4000, 4650, 5150, 6150, 7700, 8050, 12850, 16000, 22650, 29000, 38000, 50000])

    # LW
    ri_real = np.array([2.340,2.904,1.748,1.508,1.911,1.822,2.917,1.557,1.242,1.447,1.432,1.473,1.495,1.5,1.5,1.51])  # Real
    ri_imag = np.array([0.7,0.857,0.462,0.263,0.319,0.26,0.65,0.373,0.093, 0.105,0.061,0.0245,0.011,0.008,0.0068,0.018])  # Imaginary

    ri_lw = ri_real + 1j*ri_imag
    ri_sw = np.array((1.55 + 1j * 10**-3,)*(len(wn_stag_sw)-3))  # drop first 3 values due to SW & LW overlap
    wn_stag_lw_sw = np.concatenate((wn_stag_lw, wn_stag_sw[3:]))

    ri_ds = xr.Dataset(
        data_vars=dict(
            ri=(['wavenumber'], np.concatenate((ri_lw, ri_sw))),
        ),
        coords=dict(
            wavenumber_stag=(['wavenumber_stag', ], wn_stag_lw_sw),
            # wavenumber=(['wavenumber', ], wn_stag_lw_sw.rolling(lev=2).mean()),
            # wavelength=(['wavenumber', ], rayleigh_od_da.wavelength.data),
        ),
        attrs=dict(description="Dust Refractive Index"),
    )

    ri_ds['wavenumber'] = ri_ds.wavenumber_stag.rolling(wavenumber_stag=2).mean().dropna('wavenumber_stag').data
    ri_ds['wavelength'] = 10**4 / ri_ds['wavenumber']
    # ri_ds = ri_ds.set_coords('wavelength')

    #make it backward compatible for a while
    # ri_ds['wl'] = ri_ds['wavelength']

    return ri_ds

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


def mix_refractive_index(xr_in, chem_opt, wavelengths=None):
    '''
    WRF specific implementation
    Compute the volume-weighted refractive for the entire column

    wavelengths: in um
    '''

    if wavelengths is None:
        wavelengths = np.array([0.3, 0.4, 0.6, 0.999])  # default WRF wavelengths in SW

    aerosols_keys_wet = get_aerosols_keys(chem_opt, wet=True)
    aerosols_stack, dummy = get_aerosols_pm_stack(xr_in, aerosols_keys_wet)  # , pm_sizes=[d_min, d_max], combine_modes=True, combine_organics=True, combine_sea_salt=True, convert_to_stp=True)

    # the weight will be aerosols mass, integrate vertically
    z_stag = (xr_in['PH'] + xr_in['PHB']) / 9.8
    dz = z_stag.diff(dim='bottom_top_stag').rename({'bottom_top_stag': 'bottom_top'})  # m
    weights = (aerosols_stack*dz).sum(dim='bottom_top')  # ug / m^2

    ri_vos = []
    ris = []
    for aerosol_key in aerosols_keys_wet:
        key = get_molecule_key_from_aerosol_key(aerosol_key)
        ri = get_spectral_refractive_index(key, wavelengths)
        ri_vos.append(ri)
        ris.append(ri['ri'])

    ris = xr.DataArray(data=np.array(ris), dims=['aerosol', 'wavelength'])
    ris['aerosol'] = weights.aerosol
    ris['wavelength'] = wavelengths

    volume_weighted_ri_ds = ris.weighted(weights).mean(dim='aerosol').to_dataset(name='ri')
    # volume_weighted_ri_ds['wavelength'] = wavelengths

    return volume_weighted_ri_ds
import numpy as np
import disort
import pandas as pd
from scipy import special
import xarray as xr
import pvlib
import scipy as sp
from climpy.utils.file_path_utils import get_root_storage_path_on_hpc

'''
The meteorologically significant spectral range extends from 300 nm to 3000nm (short-wave radiation). Approximately 96% of the complete extra-terrestrial radiation is situated within this spectral range.
'''

print('TODO: compute_phase_function_moments needs to be tested, especially with aerosols')

RRTM_SW_WN_RANGE = [820, 50000]  # cm-1
RRTM_LW_WN_RANGE = [10, 3250]  # cm-1
RRTM_SW_LW_WN_RANGE = [RRTM_LW_WN_RANGE[0], RRTM_SW_WN_RANGE[1]]

class DisortSetup(object):
    '''
    Resembles the DISORT settings preserving the input naming
    '''

    def __init__(self, NSTR=16):
        self.HEADER = 'pyDISORT header'
        self.NSTR = NSTR
        self.NMOM = NSTR + 1
        self.DO_PSEUDO_SPHERE = True
        self.DELTAMPLUS = True  # requires NMOM = NSTR + 1

        self.USRANG = False
        self.NUMU = NSTR
        self.UMU = None

        self.NPHI = 1
        self.PHI = np.zeros((self.NPHI,))

        self.PLANK = False
        self.ONLYFL = False
        self.wn_grid_step = 1  # width of the wavenumber range in cm^-1

        self.TEMIS = 0  # TOA emissivity for Thernal


def run_disort_spectral(op_ds, atm_stag_ds, disort_setup_vo, adaptive_thermal_emission=True):
    '''
    adaptive_thermal_emission: means that PLANK is turned on depending on the wl for the LW spectrum
    '''
    def get_toa_spectral_irradiance():
        '''
        Options are AER or MODTRAN (https://www.nrel.gov/grid/solar-resource/spectra.html)
        '''
        sun_spectral_irradiance_df = get_aer_solar_constant(in_wavelength_grid=False)  # prep_chanceetal_sun_spectral_irradiance()
        # aer spans entire solar range. So, use 0 outside the coverage
        f = sp.interpolate.interp1d(sun_spectral_irradiance_df.index, sun_spectral_irradiance_df['irradiance'], bounds_error=False, fill_value=0)
        irradiance_I = f(op_ds.wavenumber)
        sun_spectral_irradiance_df = pd.Series(irradiance_I, index=op_ds.wavenumber)
        return sun_spectral_irradiance_df

    sun_spectral_irradiance_df = get_toa_spectral_irradiance()  # interpolate solar function to the requested wavenumber

    if adaptive_thermal_emission:
        print('Adaptive thermal emission is ON. WIll include thermal sources for all WNs <= {}'.format(RRTM_LW_WN_RANGE[1]))

    spectral_list = []
    for wn in op_ds.wavenumber.data:
        disort_setup_vo.HEADER = 'pyDISORT: WN {} out of {}'.format(wn, op_ds.wavenumber.data[-1])
        '''
        If I turn on the thermal emissions, the units will be determined by PLKAVG, which is MKS (W/m2)
        Since I don't have the control over PLKAVG units, and to avoid the mix up of the SW&LW units, I keep the FBEAM units synced    
        '''
        disort_setup_vo.FBEAM = sun_spectral_irradiance_df[sun_spectral_irradiance_df.index == wn].item()
        # apply wn range width
        disort_setup_vo.FBEAM *= disort_setup_vo.wn_grid_step  # don't forget to divide back

        if adaptive_thermal_emission and wn <= RRTM_LW_WN_RANGE[1]:  # for wavelengths > ~ 3 um
            disort_setup_vo.PLANK = True
            # disort_setup_vo.FBEAM = 0
        else:  # forcibly swithced off otherwise
            disort_setup_vo.PLANK = False
        disort_output_ds = run_disort(op_ds.sel(wavenumber=wn), atm_stag_ds, disort_setup_vo)
        for key in disort_output_ds.keys():
            disort_output_ds[key] /= disort_setup_vo.wn_grid_step  # convert flux in W/m^2 to W/m^2/cm^-1
        spectral_list += [disort_output_ds, ]

    disort_spectral_output_ds = xr.concat(spectral_list, dim='wavenumber')

    return disort_spectral_output_ds


def run_disort(op_rho_ds, atm_stag_ds, disort_setup_vo):
    '''
    Monochromatic version (single wavelength)
    op_rho_ds is the LBLRTM output on the RHO grid
    atm_stag_ds is (usually MERRA2) atmospheric profile on STAGGERED grid
    '''

    NLYR = op_rho_ds.dims['level']
    DTAUC = op_rho_ds.od
    SSALB = op_rho_ds.ssa

    if np.any(np.isnan(DTAUC * SSALB)):
        raise Exception('DisortController:run_disort, optical depth or ssa has NaN values')

    NSTR = disort_setup_vo.NSTR
    NMOM = disort_setup_vo.NMOM

    if NMOM < NSTR:
        raise Exception('DisortController:run_disort, NMOM<NSTR, increase NMOM')
        NMOM = NSTR

    pmoms = ()
    for momentIndex in range(NMOM + 1):
        pmom_item = compute_phase_function_moments(op_rho_ds.phase_function, momentIndex)
        pmoms += (pmom_item,)
    pmom = xr.concat(pmoms, dim='phase_function_moment')
    # print('PF Moments: {}'.format(pmom[:,0]))
    # print('PF Moments: {}'.format(pmom))

    TEMPER = atm_stag_ds.t  # at stag grid
    USRTAU = False
    NTAU = NLYR + 1
    UTAU = np.zeros((NTAU,))  # Unsued (USRTAU is false), but have to initialize for F2PY dimensions logic

    USRANG = disort_setup_vo.USRANG
    NUMU = disort_setup_vo.NUMU
    UMU = disort_setup_vo.UMU
    if USRANG:  # UMU has to be initialized
        if (UMU == 0).any():  # UMU must NOT have any zero values
            raise ('pyDISORT:run_disort. UMU must not have any zero values')
    elif UMU is None:
        UMU = np.zeros((NUMU,))  # I have to have this line to initialize dimensions in DISORT right. Otherwise F2py struglles (which can be fixed probably)
    elif UMU.shape != (NUMU,):  # In this case UMU should not be specifed (since USRANG is false).
        raise Exception('disort_utils:run_disort. Incosistent UMU shape. UMU should not be specified at all')

    NPHI = disort_setup_vo.NPHI  # should be after the USRANG, but I need NPHI
    PHI = disort_setup_vo.PHI

    IBCND = 0
    FBEAM = disort_setup_vo.FBEAM
    UMU0 = np.cos(np.deg2rad(disort_setup_vo.zenith_angle_degree))  # Corresponding incident flux is UMU0 times FBEAM.
    PHI0 = disort_setup_vo.azimuth_angle_degree

    FISOT = 0

    LAMBER = True  # simple Lambertian reflectivity
    ALBEDO = disort_setup_vo.albedo

    BTEMP = atm_stag_ds.skt
    TTEMP = atm_stag_ds.t[atm_stag_ds.level == atm_stag_ds.level.min()]  # TOA or min p. Equivalent to atm_ds.t[-1]
    # TTEMP = 0
    # print('TTEMP {} '.format(TTEMP.data))
    TEMIS = disort_setup_vo.TEMIS

    PLANK = disort_setup_vo.PLANK  # turns on thermal emissions False
    '''
    Thermal emissions require a wn region for calculating Planck function.
    Assume, that DISORT works with a wn range. Keep this range equal to 1 cm^-1 to avoid extra conversion from W/m^2 to W/m^2/cm^-1.
    If you change the interval, make sure to adjust how DISORT output is treated. (especially, FBEAM & PLANK)
    '''
    WVNMLO = op_rho_ds.wavenumber - 0.5 * disort_setup_vo.wn_grid_step  # 0  # only if PLANK
    WVNMHI = op_rho_ds.wavenumber + 0.5 * disort_setup_vo.wn_grid_step  #  50000
    # print('WVNMLO {}, WVNMLO {}'.format(WVNMLO.data, WVNMHI.data))

    ACCUR = 0.0  # should be between 0 and 0.1. I used to use single(0.005)
    NTEST = 0
    NPASS = 0

    ONLYFL = disort_setup_vo.ONLYFL
    PRNT = np.array([True, True, True, True, False])  # more verbose
    # PRNT = np.array([True, False, False, False, True])  # less verbose
    PRNT = np.array([False, False, False, False, False])  # None
    HEADER = disort_setup_vo.HEADER # use '' if crashes

    # DISORT LAYERING CONVENTION:  Layers are numbered from the top boundary down.
    DTAUC = DTAUC.sel(level=slice(None, None, -1))  # reverse the order of the profiles
    SSALB = SSALB.sel(level=slice(None, None, -1))
    PMOM = pmom.sel(level=slice(None, None, -1))
    TEMPER = TEMPER.sel(level=slice(None, None, -1))
    TEMPER = TEMPER.squeeze()

    DO_PSEUDO_SPHERE = disort_setup_vo.DO_PSEUDO_SPHERE  # setting to true fixed negative radiances
    DELTAMPLUS = disort_setup_vo.DELTAMPLUS  # requires NMOM = NSTR + 1;

    # %%
    H_LYR = np.zeros([NLYR + 1, ])
    RHOQ = np.zeros((int(NSTR / 2), int(NSTR / 2) + 1, NSTR))
    RHOU = np.zeros((NUMU, int(NSTR / 2) + 1, NSTR))
    EMUST = np.zeros(NUMU)
    BEMST = np.zeros(int(NSTR / 2))
    RHO_ACCURATE = np.zeros((NUMU, NPHI))
    EARTH_RADIUS = 6371.0

    # %%
    RFLDIR, RFLDN, FLUP, DFDT, UAVG, UU, ALBMED, TRNMED = disort.disort(USRANG, USRTAU, IBCND, ONLYFL, PRNT, PLANK,
                                                                        LAMBER, DELTAMPLUS, DO_PSEUDO_SPHERE, DTAUC,
                                                                        SSALB, PMOM, TEMPER, WVNMLO, WVNMHI, UTAU, UMU0,
                                                                        PHI0, UMU, PHI, FBEAM, FISOT, ALBEDO, BTEMP,
                                                                        TTEMP, TEMIS, EARTH_RADIUS, H_LYR, RHOQ, RHOU,
                                                                        RHO_ACCURATE, BEMST, EMUST, ACCUR, HEADER)

    # %%
    '''
    Note the thermal emissions force units to W/m^2. Outside of disort I work with W/m^2/cm^-1.
    Here, I'm keeping the lenght of the wn range in LW equal to 1cm^-1, so there is no need for postprocessing.
    '''
    RFLDIR = np.flipud(RFLDIR)  # restore the layers ordering, bottom to top
    RFLDN = np.flipud(RFLDN)
    FLUP = np.flipud(FLUP)
    UAVG = np.flipud(UAVG)
    UU = np.flip(UU, axis=1)

    # new dims order: level, polar agnels, azimuthal_ange
    UU = np.swapaxes(UU, 0, 1)

    # inject the wavelength dimension
    disort_output_ds = xr.Dataset(
        data_vars=dict(
            direct_flux_down=(["level", "wavenumber", ], RFLDIR[:, np.newaxis]),
            diffuse_flux_down=(["level", "wavenumber", ], RFLDN[:, np.newaxis]),
            diffuse_flux_up=(["level", "wavenumber", ], FLUP[:, np.newaxis]),  # up is only diffuse
            #  UAVG is mean intensity, thus normalized over the entire sphere, multiply by 4*pi to get actinic flux
            actinic_flux=(["level", "wavenumber", ], 4 * np.pi * UAVG[:, np.newaxis]),
            # Do not confuse computational polar angles (cos of) with Phase Function angles in op_rho_ds
            # UMU should hold the computational angles, but they are not returned at the moment and require editing disort.pyf (f2py)
            radiances=(["level", 'radiance_cos_of_polar_angles', 'radiance_azimuthal_angle', 'wavenumber'], UU[..., np.newaxis]),
            # Azimuthal angles is in degree (PHI)
        ),
        coords=dict(
            level=(["level", ], atm_stag_ds.level.data),
            wavenumber=(["wavenumber", ], np.array((op_rho_ds.wavenumber.item(),))),  #
            wavelength=(["wavenumber", ], np.array((op_rho_ds.wavelength.item(),))),  #
            radiance_cos_of_polar_angles=(["radiance_cos_of_polar_angles", ], UMU),
            # the cosines of the computational polar angles
            radiacens_azimuthal_angle=(["radiance_azimuthal_angle", ], PHI),
            # Azimuthal output angles (in degrees) # PHI
        ),
        attrs=dict(description="pyDISORT output"),
    )

    # pp
    disort_output_ds['down_minus_up_flux'] = disort_output_ds.direct_flux_down + disort_output_ds.diffuse_flux_down - disort_output_ds.diffuse_flux_up

    return disort_output_ds


def compute_phase_function_moments(phase_function_da, n):
    '''
    expansion of phase function is given by:
    function P(u) for each band is defined as: P(u) = sum over streams l { (2l+1) (PHASE_l) (P_l(u)) }
    where
    u = cos(theta)
    PHASE_l = the lth moment of the phase function
    P_l(u) = lth Legendre polynomial,

      phaseFunction is a function of (angle, layer)
    '''

    # def legendre(n, x):
    #     res = []
    #     for m in range(n + 1):
    #         res.append(special.lpmv(m, n, x))
    #     return np.array(res)

    # associatedLegPol = legendre(n, x)
    # pmom = np.trapz(phase_function_df * associatedLegPol, x)
    x = np.cos(phase_function_da.angle)
    associatedLegPol = special.lpmv(0, n, x)
    phase_function_da['cos(angle)'] = np.cos(phase_function_da.angle)
    integrand = phase_function_da * associatedLegPol
    pmom = integrand.integrate('cos(angle)')

    pmom *= -1  # invert sign because integration is wrong way
    # don't forget coefficient in front of integral
    # pmom = (2*n+1)/2 * pmom
    # add given the specific expansion, RRTM moved the (2n+1) from the coefficient to the expansion
    pmom = 1 / 2 * pmom

    if n == 0:  # % first moment has to be 1. if the PH is entirely 0 ( I think it is unphysical), then I get zeros. Fix it
        pmom[:] = 1

    if (np.abs(pmom) > 1).any():
        raise ('compute_phase_function_moments: pmom magnitude error')

    return pmom


def setup_viewing_geomtry(disort_setup_vo, lat, lon, date):
    '''
    Sun viewing geometry. Options are pvlib and dnppy

    :param disort_setup_vo:
    :param lat:
    :param lon:
    :param date: use UTC time
    :return:
    '''

    # TODO: supply altitude
    # TODO: have to specify time zone: , tz=site.tz)
    # TODO: make it work with the set of coordinates
    solpos = pvlib.solarposition.get_solarposition(date, lat, lon, 0)

    disort_setup_vo.zenith_angle_degree = solpos['zenith'].iloc[0]
    disort_setup_vo.azimuth_angle_degree = solpos['azimuth'].iloc[0]


def prep_chanceetal_sun_spectral_irradiance():
    # https://www.sciencedirect.com/science/article/pii/S0022407310000610
    file_path = get_root_storage_path_on_hpc() + '/Data/Harvard/SAO2010_solar_spectrum/sao2010.solref.converted.txt'
    delimiter = ' '

    df = pd.read_table(file_path, skiprows=range(4), header=None, delim_whitespace=True, usecols=[0, 2], index_col=0, names=['wavelength', 'irradiance'])
    df.index *= 10 ** -3  # um  # 'wavelength'
    df['irradiance'] *= 10 ** 3  # W/(m2 um)
    return df


def get_aer_solar_constant(in_wavelength_grid=False):
    '''
    This is output from AER extract-solar software
    If WN grid, Units are W/(m2 cm-1).
    If grid is converted to WL, units are W/m2 um-1
    '''
    fp = '/Users/osipovs/PycharmProjects/solar-source-function/run_example_average_solar_constant/solar_rad_nrl3comp_820_50000'  # this is RRTM SW range
    aer_solar_df = pd.read_csv(fp, skiprows=[0, ], delim_whitespace=True, header=None, names=['wavenumber', 'irradiance'])
    if in_wavelength_grid:  # convert from WN to WL. BE careful
        wn_stag = aer_solar_df['wavenumber'].rolling(2).mean().to_numpy()
        wn_stag[0] = aer_solar_df['wavenumber'][0] - 0.005  # 0.05 is the WN step
        wn_stag = np.append(wn_stag, aer_solar_df['wavenumber'].iloc[-1] + 0.005)
        wl_stag = 10 ** 4 / wn_stag
        dwn = np.diff(wn_stag)  # step in wn
        dwl = -1 * np.diff(wl_stag)
        wl_rho = pd.Series(wl_stag).rolling(2).mean().to_numpy()[1:]  # derive rho grid
        aer_solar_df = pd.Series(aer_solar_df['irradiance'].to_numpy() * dwn / dwl, index=wl_rho, name='irradiance').to_frame()  # W/m2 um-1
        aer_solar_df = aer_solar_df[::-1]
    else:
        aer_solar_df.set_index('wavenumber', inplace=True)

    return aer_solar_df


def setup_surface_albedo(disort_setup_vo):
    disort_setup_vo.albedo = 0


def checkin_and_fix(ds):
    if ds.od.min() < 0:
        print('Min')
        print(ds.min())
        print('Max')
        print(ds.max())

        count = (ds.od < 0).sum()
        print('Replacing via interpolation {} negative OD values'.format(count.item()))
        ds['od'] = ds.od.where(ds.od > 0).interpolate_na(dim='wavenumber')

        # if np.abs(ds.od.min()) < 10**-6:
        #     count = (ds.od < 0).sum()
        #     print('Zeroing out {} negative OD values'.format(count.item()))
        #     ds.od.where(ds.od<0)[:] = 0
        #     ds['od'] = ds.od.where(ds.od>0).fillna(0)
        # else:
        #     raise Exception('Dataset contains negative values beyound the tollerance. Check what is wrong.')
    if ds.ssa.max() > 1:
        print('SSA is above 1')
        print(ds.ssa.max())
        ds['ssa'] = ds.ssa.where(ds.ssa < 1).fillna(1)
        # if ds.ssa.max() < 1 + 10**-6:
        #     ds['ssa'] = ds.ssa.where(ds.ssa < 1).fillna(1)
        # else:
        #     raise Exception('SSA > 1')
    count = (ds.ssa < 0).sum()
    if count > 0:
        print('Replacing via interpolation {} < 0 ssa values'.format(count.item()))
        ds['ssa'] = ds.ssa.where(ds.ssa > 0).fillna(0)

    if ds.od.isnull().any() or ds.ssa.isnull().any() or ds.g.isnull().any() or ds.phase_function.isnull().any():
        raise Exception('NaN values in the optical properties')

import os

import numpy as np
import scipy as sp
import scipy.constants
import xarray as xr
from ambiance import Atmosphere

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

DRY_AIR_MOLAR_MASS = 29 * 10 ** -3  # kg mol^-1


def air_number_density(p, t):
    '''
    Compute air number density using ideal law
    :param p: pressure, [Pa]
    :param t: temperature, [K]
    :return: number density, [molecules m^-3]
    '''

    # Avogadro's number is sp.constants.N_A, 6*10**23, [molecules mol^-1]
    # molar gas constant is sp.constants.R, 8.31, [J mol^-1 K^-1]

    n_a = sp.constants.N_A * p / (sp.constants.R * t)
    return n_a


def air_mass_density(p, t):
    '''
    estimate air mass density, p=rho R T or p = rho * R / M *T, where M is a molar mass

    :param p: pressure, (Pa)
    :param t: temperature, (K)
    :return: air density, (kg/m^3)
    '''

    air_rho = p / sp.constants.R / t * DRY_AIR_MOLAR_MASS

    return air_rho


def compute_column_from_vmr_profile(p, t, dz, gas_ppmv, z_dim_axis=0, in_DU=True):
    """
    Computes for a given gas profile the column loading (by default in Dobson Units (DU))
    :param p: in Pa
    :param t: in K (regular, not potential!)
    :param dz: in meters (derived from z_stag)
    :param gas_ppmv: gas profile in units of ppmv
    :param z_dim_axis:
    :return:
    """

    # TODO: replace these two lines with the decorator
    n_air = air_number_density(p, t)  # molecules / m^3
    gas_number_density = gas_ppmv * 10**-6 * n_air  # molecules / m**3

    gas_dobson_units = compute_column_from_nd_profile(dz, gas_number_density, z_dim_axis, in_DU)
    return gas_dobson_units


def compute_column_from_nd_profile(dz, gas_number_density, z_dim_axis=0, in_DU=True):
    """
    Computes for a given gas profile the column loading (possibly in Dobson Units (DU))
    :param dz: in meters (derived from z_stag)
    :param gas_number_density: gas number density profile in [molecules m^-3]
    :param z_dim_axis:
    :return: gas column (integrated verically) in [molecules m^-2] or in DU
    """

    # dont forget to convert column density from #/m^2 to #/cm^2
    gas_column = np.sum(gas_number_density * dz, axis=z_dim_axis)
    if in_DU:
        DU = 2.69 * 10 ** 20  # molecules m**-2
        gas_column /= DU

    return gas_column


def rel_humidity_to_mass_concentration(atm_stag_ds):
    # work with concentration instead of relative humidity
    from metpy.units import units
    from metpy.calc import mixing_ratio_from_relative_humidity, density

    # kg/kg
    h2o_mmr = mixing_ratio_from_relative_humidity(atm_stag_ds.p.values * units.hPa, atm_stag_ds.t.values * units.degK, atm_stag_ds.r.values/100)  # .to('g/kg')
    # kg/m^3
    air_density = density(atm_stag_ds.p.values * units.hPa, atm_stag_ds.t.values * units.degK, 0 * units('g/kg'))  # otherwise provide water mixing ration and get wet air density
    h2o_mass_concentration = h2o_mmr * air_density  # kg / m^3
    return h2o_mass_concentration  # kg / m^3


def get_saturation_vapor_pressure_Magnus_formula(temperature_C):
    '''
    Temperature in Celcius
    # https://en.wikipedia.org/wiki/Clausius%E2%80%93Clapeyron_relation#August%E2%80%93Roche%E2%80%93Magnus_formula


    # e_s = get_saturation_vapor_pressure_Magnus_formula(atm_stag_ds.t-273.15)
    # p_h2o = atm_stag_ds.r/10**2 * e_s  # h2o pressure
    # p_dry = atm_stag_ds.p-e_s  # dry air pressure

    '''

    e_s = 6.1094*np.exp(17.625*temperature_C/(temperature_C+243.04))
    return e_s  # in hPa


def get_standard_atmospheric_profile(date):

    '''
    This was done for tonga_water_rf (LBL)
    setup standard atmosphere

    :param date:
    :return:
    '''
    z_stag = np.linspace(0, 75e3, num=100)
    standard_atm_stag_ds = Atmosphere(z_stag)

    # get the water from MERRA2 output
    ds = xr.open_dataset('/Users/osipovs/Data/NASA/GMI/2014/gmic_MERRA_2014_jan.amonthly.nc')
    ds = ds.rename({'longitude_dim': 'lon', 'latitude_dim':'lat', 'eta_dim':'level', 'kel':'t'})
    ds = ds.isel(rec_dim=0)
    ds = ds.sel(lat=lats, lon=lons, method='nearest')  # do the selection

    rh = 10**2 * relative_humidity_from_mixing_ratio(ds.level * units.hPa, ds.t * units.degK, ds.metwater * 18/29)  # %
    rh = rh.where(rh>=0, 10**-6)  # stupid saturation_mixing_ratio produces negative values
    rh = rh.interp(level=standard_atm_stag_ds.pressure/10**2, kwargs={"fill_value": "extrapolate"})  # TODO: extrapolating blindly is bad

    atm_stag_ds = xr.Dataset(
        data_vars=dict(
            t=(["level",], standard_atm_stag_ds.temperature),
            p=(["level",], standard_atm_stag_ds.pressure/10**2),
            z=(["level",], z_stag),
            r=(["level",], rh.squeeze().data, dict(units='relative humidity')),  # vmr (ppmv)
        ),
        coords=dict(
            level=(["level", ], standard_atm_stag_ds.pressure/10**2),
        ),
        attrs=dict(description="International Civil Aviation Organization ; Manual Of The ICAO Standard Atmosphere – 3rd Edition 1993 (Doc 7488) – extended to 80 kilometres (262 500 feet)"),
    )


    atm_stag_ds = atm_stag_ds.expand_dims(dim={'lat': 1, 'lon':1}, axis=[1,2])
    atm_stag_ds['skt'] = standard_atm_stag_ds.temperature[0]

    return atm_stag_ds


def get_atmospheric_profile(date):
    '''
    Sets up the profile for LBL calculations
    date is needed to build the file path
    The profile is suitable for LBLRTM & DISORT calculations
    '''
    date_str = date.strftime('%Y%m%d')
    fp = '{}sfc/{}'.format('/work/mm0062/b302074/Data/ECMWF/EraInterim/netcdf/global/F128/', 'ECMWF_sfc_{}_{}.nc'.format(date_str, date_str))
    fp = os.path.expanduser('~') + '/Temp/ECMWF_sfc_{}_{}.nc'.format(date_str, date_str)  # temp local
    sfc_ds = xr.open_dataset(fp)
    sfc_ds = sfc_ds.rename_vars({'z': 'z_sfc'})
    fp = '{}pl/{}'.format('/work/mm0062/b302074/Data/ECMWF/EraInterim/netcdf/global/F128/', 'ECMWF_pl_{}_{}.nc'.format(date_str, date_str))
    fp = os.path.expanduser('~') + '/Temp/ECMWF_pl_{}_{}.nc'.format(date_str, date_str)  # temp local
    profile_ds = xr.open_dataset(fp)

    # add pressure variable
    profile_ds['p'] = (('level', ), profile_ds.level.data)  # keep in 1D although general approach is N-D

    # reverse the z direction to have indexing start at the surface
    profile_ds = profile_ds.sel(level=slice(None, None, -1))

    # merge surface and profile datasets
    ds = xr.merge([sfc_ds, profile_ds])
    ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})

    # correct few things
    ds.sp[:] /= 10**2
    ds.sp.attrs['units'] = 'hPa'

    # convert geopotential to height in meters
    # TODO: it is probably faster to do it after location sampling or in output parser
    g = 9.8  # m/sec**2
    ds.z_sfc[:] /= g
    ds.z_sfc.attrs['units']='m'
    ds.z_sfc.attrs['long_name'] = 'Height'
    ds.z_sfc.attrs['standard_name'] = 'height'

    ds.z[:] /= g
    ds.z.attrs['units'] = 'm'
    ds.z.attrs['long_name'] = 'Height'
    ds.z.attrs['standard_name'] = 'height'

    # make sure there no negative values in RH. MERRA2 may have them
    # if (ds.r<0).any():
    #     raise Exception('get_atmospheric_profile: negative values in the relative humidity profile')

    return ds

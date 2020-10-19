import copy
import datetime as dt

import netCDF4
import numpy as np
import glob

from climpy.utils import mie_utils as mie
from climpy.utils.file_path_utils import get_root_storage_path_on_hpc, convert_file_path_mask_to_list
from natsort.natsort import natsorted
from dateutil.relativedelta import relativedelta

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

from climpy.utils.refractive_index_utils import get_Williams_Palmer_refractive_index


def prepare_sparc_asap_stratospheric_optical_depth():
    txt_file_path = get_root_storage_path_on_hpc() + 'Data/SPARC/ASAP/SAGE_1020_OD_Filled.dat'
    data = np.loadtxt(txt_file_path, skiprows=2)

    lat_only_cols = np.arange(1, data.shape[1])
    lat_data = np.loadtxt(txt_file_path, skiprows=1, usecols=lat_only_cols)
    lat_data = lat_data[0]

    # derive proper time data
    fractional_time_data = data[:,0]
    year = np.floor(fractional_time_data)
    time_data = np.empty(data.shape[0], dtype=dt.datetime)
    for i in range(len(year)):
        boy = dt.datetime(int(year[i]),1,1)
        eoy = dt.datetime(int(year[i]+1), 1, 1)
        seconds = (fractional_time_data[i]-boy.year) * (eoy-boy).total_seconds()

        time_data[i] = boy+dt.timedelta(seconds=seconds)

    aod_data = data[:,1:]
    aod_data[aod_data == 9.999] = np.NaN
    # data in the ASAP is saved as log 10 of aod, convert back
    aod_data = 10**aod_data

    vo = {}
    vo['data'] = aod_data
    vo['time'] = time_data
    vo['lat'] = lat_data
    return vo


def prepare_sparc_asap_profile_data(is_filled=True, filter_unphysical_data=True):
    """

    :param is_filled:
    :return: field 'data' contains extinction (not AOD)
    """

    if not is_filled:
        # when data is not filled, some of the lat bands are missing and has to be read in manually
        raise ValueError('is_filled=False is not implemented yet')

    # read asap vertically resolved data
    file_path_mask = get_root_storage_path_on_hpc() + '/Data/SPARC/ASAP/SAGE_NoInterp_Data/*.dat'
    n_params = 26

    if is_filled:
        file_path_mask = get_root_storage_path_on_hpc() + '/Data/SPARC/ASAP/SAGE_Filled_Data/*.dat'
        n_params = 28

    files_list = convert_file_path_mask_to_list(file_path_mask)

    data = np.zeros((len(files_list), 32, 70, n_params))
    for time_index in range(data.shape[0]):
        current_file_path = files_list[time_index]
        f = open(current_file_path)
        # loop through the latitudes
        for i in range(32):
            # headers
            line = f.readline()
            line = f.readline()
            line = f.readline()
            line = f.readline()
            line = f.readline()
            line = f.readline()
            header1 = f.readline()
            header2 = f.readline()
            # data
            # loop through the altitude
            for j in range(70):
                line = f.readline()
                data[time_index, i, j] = np.fromstring(line, sep=' ')

        f.close()

    # missing values are -1
    data[data == -1] = np.NAN
    # sometimes there and Inf values
    data[data == np.inf] = np.NAN

    if is_filled:
        # replace zero temperature and pressure
        ind = data == 0
        ind[:,:,:,1:24] = 0
        ind[:, :, :, 27] = 0
        data[ind] = np.NAN
        data[ind] = np.NAN

    # generate time data
    time_data = np.empty((data.shape[0],), dtype=object)
    for time_index in range(data.shape[0]):
        time_data[time_index] = dt.datetime(1984, 10,15) + relativedelta(months=+time_index)

    # generate lat data
    lat_boundaries_data = np.arange(-80, 85, 5)
    lat_data = (lat_boundaries_data[1:] + lat_boundaries_data[:-1])/2

    altitude_data = data[0,0,:,0]

    vo = {}
    vo['data'] = data
    vo['time'] = time_data
    vo['altitude'] = altitude_data
    vo['lat'] = lat_data
    vo['lat_stag'] = lat_boundaries_data

    vo['header1'] = header1
    vo['header2'] = header2

    # PP

    # make the lat shape match the data shape
    lat = vo['lat']
    # replicate weight for each time snapshot
    lat = np.repeat(lat[np.newaxis, :], vo['data'].shape[0], axis=0)
    lat = np.repeat(lat[..., np.newaxis], vo['altitude'].shape[0], axis=-1)
    vo['lat'] = lat

    if filter_unphysical_data:
        r_eff_ind = -7
        r_eff = vo['data'][..., r_eff_ind]  # microns
        # filter unrealistic values
        ind = r_eff > 10 ** 2
        r_eff[ind] = np.NaN
        print('{} r_eff > 10**2 were removed'.format(np.sum(ind)))
        ind = r_eff < 10 ** -2
        r_eff[ind] = np.NaN
        print('{} r_eff < 10**-2 were removed'.format(np.sum(ind)))
        vo['data'][..., r_eff_ind] = r_eff

        ext_1020 = vo['data'][..., 1] * 10 ** -3  # m^-1
        # filter unrealistic values
        ind = ext_1020 > 10 ** -2
        ext_1020[ind] = np.NaN
        print('{} ext_1020 > 10**-2 were removed'.format(np.sum(ind)))
        vo['data'][..., 1] = ext_1020

        #TODO filter 525 nm too
        # ext_525 = sparc_profile_vo['data'][..., 5] * 10 ** -3  # m^-1

    return vo


def disassemble_sparc_into_diags(sparc_profile_vo):
    '''
    var_index: 1 is 1020 nm ext, -7 is r_eff
    '''

    r_eff_vo = copy.deepcopy(sparc_profile_vo)
    r_eff_vo['data'] = sparc_profile_vo['data'][..., -7]  # microns

    ext_1020_vo = copy.deepcopy(sparc_profile_vo)
    ext_1020_vo['data'] = sparc_profile_vo['data'][..., 1]  # microns

    # ext_525_vo = copy.deepcopy(sparc_profile_vo)
    # ext_525_vo['data'] = sparc_profile_vo['data'][..., 5]  # microns

    # pp, derive column AOD
    dz = 0.5 * 10**3  # m
    aod_1020_vo = copy.deepcopy(sparc_profile_vo)
    aod_1020_vo['data'] = np.nansum(sparc_profile_vo['data'][:, :, :, 1] * dz, axis=2)
    aod_1020_vo['lat'] = sparc_profile_vo['lat'][:,:,0]

    return r_eff_vo, ext_1020_vo, aod_1020_vo


def prepare_sato_data():
    # this one is only up to 2000
    sato_nc_file_path = get_root_storage_path_on_hpc() + '/Data/GISS/Volcanoes/STRATAER.VOL.CROWLEY-SATO.800-2010_hdr.nc'
    # this one is actually up to 2010
    sato_nc_file_path = get_root_storage_path_on_hpc() + '/Data/GISS/Volcanoes/STRATAER.VOL.GAO-SATO.850-2010_v2hdr.nc'

    nc = netCDF4.Dataset(sato_nc_file_path)
    rawTimeData = nc.variables['date']

    time_data = np.zeros(rawTimeData.shape, dtype=dt.datetime)
    for i in range(rawTimeData.size):
        year = int(np.floor(rawTimeData[i]))
        yearDays = (dt.datetime(year + 1, 1, 1) - dt.datetime(year, 1, 1)).days
        day = int((rawTimeData[i] - year) * yearDays)
        time_data[i] = dt.datetime(year, 1, 1) + dt.timedelta(days=day)

    vo = {}
    vo['aod'] = nc.variables['tauALL'] # 0.55 um
    vo['time'] = time_data
    vo['lat'] = nc.variables['lat']

    return vo


def prepare_ammann_data():
    nc_fp = get_root_storage_path_on_hpc() + '/Data/VolcanicDataSet/Ammann/ammann2003b_volcanics.nc'

    nc = netCDF4.Dataset(nc_fp)
    vo = {}
    vo['aod'] = nc.variables['TAUSTR'] # 0.55 um
    vo['time'] = [dt.datetime.strptime(str(date_item), '%Y%m') for date_item in nc.variables['time'][:]]
    vo['lat'] = nc.variables['lat']

    return vo


def prepare_cmip6_data():
    cmip6_fp = "/shaheen/project/k1090/predybe/CM2.1/home/CM2.1_E1_Pinatubo/CM2.1U_Control-1990_IC/pinatubo_aerosols/pinatuboCMIP6/extsw_data.nc"
    nc = netCDF4.Dataset(cmip6_fp)

    vo = {}
    vo['extsw'] = np.squeeze(nc.variables['extsw_b06'][:]) #0.55 um
    vo['p_rho'] = nc.variables['pfull'][:]#hPa
    vo['lat'] = nc.variables['lat'][:]
    vo['lon'] = nc.variables['lon'][:]
    vo['time'] = netCDF4.num2date(nc.variables['time'][:], nc.variables['time'].units)

    return vo


def derive_sparc_so4_wet_mass(r_eff_vo, ext_1020_vo):
    '''
    Derive the SO4 mass that corresponds to the Effective radius and extinction at 1020 nm
    The Mass is wet and corresponds to the 75% solution
    '''

    r_eff = r_eff_vo['data']  # microns
    ext_1020 = ext_1020_vo['data']  # m^-1
    dz = 0.5  # km

    # internal grids
    dp = np.logspace(-9, -4, 40)
    r_data = dp/2  # m
    wavelengths = np.array([0.525, 1.02])  # um

    # Prepare RI
    ri_vo = get_Williams_Palmer_refractive_index()
    # interpolate RI onto internal wavelength grid
    ri = np.interp(wavelengths, ri_vo['wl'][::-1], ri_vo['ri'][::-1])  # ri = real+1j*imag
    ri_vo['ri'] = ri
    ri_vo['wl'] = wavelengths

    # Compute Mie extinction coefficients
    mie_vo = mie.get_mie_efficiencies(ri_vo['ri'], r_data*10**6, ri_vo['wl'])

    # check Mie
    # plt.ion()
    # plt.figure()
    # plt.plot(mie_vo['r_data'], mie_vo['qext'][0], '-o')
    # plt.ylabel('qext')
    # plt.xscale('log')

    # sample the SD
    # sg, dg, moment3, moment0
    sg = np.ones(r_eff.shape) * 1.8  # fix the parameter
    # r_e = r_g * exp(5/2 ln(sg)^2)
    dg = 2 * r_eff / np.exp(5/2*np.log(sg)**2)
    dg *= 10**-6  # um -> m

    # sample the SD, normalized to 1
    # THIS one is faster then sp.stats.lognorm
    dNdlogp = 1/((2*np.pi)**(1/2) * np.log(sg[..., np.newaxis])) * np.exp(-1/2 * (np.log(dp)-np.log(dg[..., np.newaxis]))**2 / np.log(sg[..., np.newaxis])**2)

    # check SD
    # time_ind = 80
    # plt.ion()
    # plt.figure()
    # plt.plot(dp, dNdlogp[time_ind, 16, 45], '-o')
    # plt.ylabel('qext')
    # plt.xscale('log')

    # Compute the OD
    dNdlogr = dNdlogp/2
    cross_section_area_transform = np.pi * r_data ** 2
    ext = np.trapz(dNdlogr[..., np.newaxis, :] * cross_section_area_transform * mie_vo['qext'], np.log(r_data), axis=-1)
    # ext *= 10**-18  # remember the units of the pdf (?, um) and cross section area (um^2)

    # check OD
    # plt.ion()
    # plt.figure()
    # plt.cla()
    # x_coord = sparc_profile_vo['time']
    # plt.plot(x_coord, ext[:, 16, 45, 1], '-o', label='Mie, unscalled')  # time, lat, alt, wl
    # plt.plot(x_coord, ext_1020[:, 16, 45]*10**-7, '-*', label='SPARC ASAP 1020')  # time, lat, alt, wl
    # # plt.plot(x_coord, moment0[:, 16, 45]*10**-21, '-*', label='moment0')
    # plt.ylabel('Extinction')
    # plt.yscale('log')
    # plt.legend()

    # scale the number of particles (m0) to match the OD in the dataset
    moment0 = ext_1020 / ext[..., 1]  # number / m^3
    # mk = m0 * dg**k * exp(k^2/2 ln(sg)^2)
    moment3 = moment0 * dg**3 * np.exp(9/2*np.log(sg)**2)  # m^3 / m^3
    # derive aerosol mass
    earth_radius = 6371*10**3  # m
    deg2m = 2*np.pi*earth_radius / 360  # about 111 km
    dlat = 5 * deg2m * np.ones(moment3.shape[1:3])  # 5 is a lat step, # sparc_profile_vo['lat'][1:]-sparc_profile_vo['lat'][:-1]
    dlon = 360 * deg2m * np.cos(np.deg2rad(ext_1020_vo['lat']))  # this is lon sum
    dlon = dlon[0]
    cell_volume = dlat*dlon*dz*10**3  # m^3
    rho_so4 = 1800  # kg m^-3
    mass = rho_so4 * np.pi/6 * moment3 * cell_volume  # kg

    so4_mass_vo = copy.deepcopy(r_eff_vo)
    so4_mass_vo['data'] = mass

    return so4_mass_vo
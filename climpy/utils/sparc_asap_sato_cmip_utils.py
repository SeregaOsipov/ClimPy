import datetime as dt

import netCDF4
import numpy as np
import glob

from climpy.utils.file_path_utils import get_root_storage_path_on_hpc, convert_file_path_mask_to_list
from natsort.natsort import natsorted
from dateutil.relativedelta import relativedelta

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'


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


def prepare_sparc_asap_profile_data(is_filled=True):
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

    return vo


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
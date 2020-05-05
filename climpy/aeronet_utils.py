from datetime import datetime, timedelta
import numpy as np
from matplotlib import mlab
import os.path
import glob
import pandas as pd

from Papers.AQABA.aqaba_utils import normalize_size_distribution
from libs.diag_decorators import time_interval_selection
from libs.file_path_utils import get_root_storage_path_on_hpc

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'


# station constants
ALL_STATIONS = '*'

# temporal resolution constants
ALL_POINTS = 'all_points'
DAILY = 'daily'
SERIS = 'series'


def get_stations_file_path(station_mask, level, res, inv):
    '''

    :param station_mask: such as *KAUST*
    :param level: 10 15 or 20
    :param res: temporal resolution, ALL_POINTS for example
    :return:
    '''

    AOD_POSTFIX = '/AOD/AOD{}'.format(level)
    INV_POSTFIX = '/INV/LEV{}/ALL'.format(level)

    postfix = AOD_POSTFIX
    if inv:
        postfix = INV_POSTFIX

    search_folder = get_root_storage_path_on_hpc() + '/Data/NASA/Aeronet/v3' + postfix + '/{}/'.format(res.upper())
    files_list = glob.glob(search_folder + station_mask)

    return files_list


def get_station_file_path(station_mask, level, res, inv):
    '''

    :param station_mask: such as *KAUST*
    :param level: 10 15 or 20
    :param res: temporal resolution, ALL_POINTS for example
    :return:
    '''

    files_list = get_station_file_path(station_mask, level, res, inv)
    if len(files_list) > 1:
        raise Exception('Aeronet: found more than one harbor that match {} in the {}'.format(station_mask, search_folder))
    if len(files_list) == 0:
        raise Exception('Aeronet: can not find harbor that match {} in the {}'.format(station_mask, search_folder))

    return files_list[0]


def get_maritime_file_path(cruise, level, res):
    '''
    build the file path to the microtops measurements (maritime aerosols section of the aeronet)
    :param cruise: such as Kommandor_Iona_17
    :param res: temporal resolution such as ALL_POINTS
    :return:
    '''

    file_path = get_root_storage_path_on_hpc() + '/Data/NASA/Aeronet/Maritime/{}/AOD/{}_{}.lev{}'.format(cruise, cruise, res, level)
    return file_path


def read_aeronet(aeronet_fp, inv=False, only_head=False):
    '''
    Generic. Reads and parses entire Aeronet file
    :param aeronet_fp:
    :param inv: TODO: it is possible to figure out INV or AOD type from the file itself
    :param only_head: is usefully to get only metadata from the file.
    :return:
    '''

    time_cols = [0, 1]
    if inv:
        time_cols = [1, 2]

    nrows = None
    if only_head:
        nrows = 1

    dateparse = lambda x: datetime.strptime(x, "%d:%m:%Y %H:%M:%S")
    aeronet_df = pd.read_csv(aeronet_fp, skiprows=6, nrows=nrows,
                             na_values=['N/A', -999],
                             parse_dates={'time': time_cols}, date_parser=dateparse)

    aeronet_df = aeronet_df.set_index('time')#, inplace=True)

    # Drop any rows that are all NaN and any cols that are all NaN
    # & then sort by the index
    # an = (aeronet.dropna(axis=1, how='all').dropna(axis=0, how='all').sort_index())

    return aeronet_df


def read_aeronet_maritime(aeronet_fp):
    '''
    reads an Aeronet Maritime file
    :param aeronet_fp:
    :param skiprows: 6 is default for Aeronet, 5 is for maritime
    :return:
    '''
    dateparse = lambda x: datetime.strptime(x, "%d:%m:%Y %H:%M:%S")
    aeronet_df = pd.read_csv(aeronet_fp, skiprows=4, na_values=['N/A'],  # N/A is probably -999.000000
                          parse_dates={'time': [0, 1]},
                          date_parser=dateparse)

    aeronet_df = aeronet_df.set_index('time')#, inplace=True)

    # Drop any rows that are all NaN and any cols that are all NaN
    # & then sort by the index
    # an = (aeronet.dropna(axis=1, how='all').dropna(axis=0, how='all').sort_index())

    return aeronet_df


def get_aod_product(station, level, res):
    aeronet_fp = get_station_file_path(station, level, res, inv=False)
    aeronet_df = read_aeronet(aeronet_fp)

    return aeronet_df


def get_inversion_product(station, level, res):
    aeronet_fp = get_station_file_path(station, level, res, inv=True)
    aeronet_df = read_aeronet(aeronet_fp, inv=True)

    return aeronet_df


def get_maritime_product(cruise, level, res):
    aeronet_fp = get_maritime_file_path(cruise, level, res)
    aeronet_df = read_aeronet_maritime(aeronet_fp)

    return aeronet_df


# more specific routines


@time_interval_selection
def get_aod_diag(station, var_key, level, res):
    aeronet_df = get_aod_product(station, level, res)

    vo = {}
    vo['data'] = aeronet_df[var_key].to_numpy()
    vo['time'] = aeronet_df.index.to_pydatetime()

    return vo


@time_interval_selection
@normalize_size_distribution
def get_size_distribution(station, level, res):
    """

    :param station:
    :param region:
    :return: dV/dlnr [µm^3/µm^2]
    """
    aeronet_df = get_inversion_product(station, level, res)

    # get columns for size distribution between 0.05 and 15 um
    si = aeronet_df.keys().get_loc('0.050000')
    en = aeronet_df.keys().get_loc('15.000000')+1
    aeronet_sd = aeronet_df.iloc[:, si:en]  # dV(r)/dlnr [µm^3/µm^2]

    aeronet_vo = {}
    aeronet_vo['data'] = aeronet_sd.to_numpy()
    aeronet_vo['time'] = aeronet_df.index.to_pydatetime()
    aeronet_vo['radii'] = aeronet_sd.columns.to_numpy(dtype=np.float) # um

    return aeronet_vo

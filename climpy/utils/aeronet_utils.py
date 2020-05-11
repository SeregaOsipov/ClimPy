import datetime as dt
from datetime import datetime
import numpy as np
from dateutil import relativedelta
import os.path
import glob
import pandas as pd
from shapely.geometry import Point

from climpy.utils.diag_decorators import time_interval_selection, normalize_size_distribution

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'


# station constants
ALL_STATIONS = '*'

# temporal resolution constants
ALL_POINTS = 'all_points'
DAILY = 'daily'
SERIS = 'series'


# YOU HAVE TO UPDATE FILE PATH, before using the module
DATA_FILE_PATH_ROOT = os.path.expanduser('~') + '/Data/NASA/Aeronet/'


def get_all_stations_and_coordinates():
    """
    get list of all Aeronet stations and their coordinates from the WEB
    :return:
    """
    stations_url = 'https://aeronet.gsfc.nasa.gov/aeronet_locations_v3.txt'
    import pandas as pd
    stations = pd.read_csv(stations_url, skiprows=1)
    return stations


def get_available_stations(time_range, aod_level):
    """
    returns the list of stations that have the time coverage for a specified level of AOD product.

    Aeronet has a list of stations for each year and mask of availability for each month.
    See "All lists" link on Aeronet website https://aeronet.gsfc.nasa.gov/Site_Lists_V3/site_index.html

    The idea is to get big table with all stations for all the years, and then filter those that have temporal coverage.

    :param time_range: [start_date end_date], datetime objects
    :param aod_level: 10, 15 or 20
    :return:
    """

    unique_years = np.arange(time_range[0].year, time_range[1].year + 1)
    year = unique_years[0]

    stations_soup = []
    for year in unique_years:
        stations_url = 'https://aeronet.gsfc.nasa.gov/Site_Lists_V3/aeronet_locations_v3_{}_lev{}.txt'.format(year, aod_level)
        stations = pd.read_csv(stations_url, skiprows=1)
        # add year information
        stations['YEAR'] = pd.Series(year, index=stations.index)
        stations_soup.append(stations)

    stations_soup = pd.concat(stations_soup)

    # now loop through the stations and check the temporal coverage
    unique_names = stations_soup['Site_Name'].unique()
    results = []
    for name in unique_names:
        subset = stations_soup[stations_soup['Site_Name'] == name]

        available_dates = pd.DatetimeIndex([])
        for index, row in subset.iterrows():
            year = row['YEAR']
            months = pd.date_range(start='{}-01-01'.format(year), periods=12, freq='MS')
            available_dates = available_dates.append(months[row['JAN':'DEC'].to_numpy(dtype=bool)])

        # round up the interval's dates because Aeronet has monthly availability dates
        start_date = dt.datetime(time_range[0].year, time_range[0].month, 1)
        end_date = dt.datetime(time_range[1].year, time_range[1].month, 1) + relativedelta.relativedelta(months=1)
        is_covered = np.any(np.logical_and(available_dates >= start_date, available_dates < end_date))
        if is_covered:
            # results.append(name)
            results.append(subset.iloc[0, 0:4])  # save the subset with the metadata


    return pd.DataFrame(results)


def filter_available_stations(domain, time_range, aod_level):
    """
    Return list of stations within the domain and time interval
    :param domain:
    :param time_range:
    :param aod_level:
    :return:
    """
    # list of stations, that have the temporal coverage
    prelim_stations = get_available_stations(time_range, aod_level)

    # filter stations within the domain
    stations_inside_domain = pd.DataFrame(columns=prelim_stations.columns)
    for index, station in prelim_stations.iterrows():
        point = Point(station['Longitude(decimal_degrees)'], station['Latitude(decimal_degrees)'])
        # TODO: this test does not account for the map projection distortions
        if domain.contains(point):
            stations_inside_domain = stations_inside_domain.append(station, ignore_index=True)

    # overview
    return stations_inside_domain


def get_stations_file_path(station_mask, level, res, inv):
    '''

    :param station_mask: such as *KAUST*
    :param level: 10 15 or 20
    :param res: temporal resolution, ALL_POINTS for example
    :return:
    '''

    AOD_PRODUCT_PATH = 'AOD/AOD{}'.format(level)
    INV_PRODUCT_PATH = 'INV/LEV{}/ALL'.format(level)

    product_path = AOD_PRODUCT_PATH
    if inv:
        product_path = INV_PRODUCT_PATH

    search_folder = DATA_FILE_PATH_ROOT + 'v3/{}/{}/'.format(product_path, res.upper())
    files_list = glob.glob(search_folder + station_mask)

    return files_list


def get_station_file_path(station_mask, level, res, inv):
    '''

    :param station_mask: such as *KAUST*
    :param level: 10 15 or 20
    :param res: temporal resolution, ALL_POINTS for example
    :return:
    '''

    files_list = get_stations_file_path(station_mask, level, res, inv)
    if len(files_list) > 1:
        raise Exception('Aeronet: found more than one harbor that match {} in the {}'.format(station_mask))
    if len(files_list) == 0:
        raise Exception('Aeronet: can not find harbor that match {} in the {}'.format(station_mask))

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


def filter_stations(file_paths, filter_impl):
    """
    Loops through the stations, reads only the first row of data and applies filter
    :param file_paths: list of file paths to process
    :param filter_impl: function, which returns true or false. You should write your own version.
    :return:
    """
    fps = []
    for file_path in file_paths:
        # print(file_path)
        if filter_impl(file_path):  # make sure that filter is computationally efficient
            fps.append(file_path)
    return fps


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

    # TODO: Drop any rows that are all NaN and any cols that are all NaN
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
    # add some meta data

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


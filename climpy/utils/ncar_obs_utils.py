import datetime as dt

import numpy as np
import pandas as pd
import scipy as sp

'''
This script is to support the NCAR airplane observations in Saudi Arabia, which focused on could seeding.
'''


#%% processed (by Duncan) size distributions


def get_processed_dust_size_distribution_in_abha():
    # prep SD aux data
    fp  = '/home/osipovs/Downloads/Abha/ASDprocessed/ASD_bins.csv'
    sd_bins_df = pd.read_csv(fp, header=0, skipinitialspace=True, sep=';')
    sd_bins_df = sd_bins_df[['ASD', 'Mid size']]
    # prep SD data
    fp  = '/home/osipovs/Downloads/Abha/ASDprocessed/RF08_20090811_ASD_no_smooth.csv'
    df = pd.read_csv(fp, header=0, skipinitialspace=True)
    # drop regions contaminated by clouds (cloud=1)
    df = df[df['cloud'] == 0]
    df = df.sort_values('Alt')
    dz = df.Alt.diff()

    # integrate vertically
    sd_columns = [column for column in df.columns if 'ASD' in column]
    # ASD: aerosol conc in each size bin dN/dlogDp (number/cm^3); aerosol size in micrometers
    # ALT: altitude in meters
    column_sd_df = df[sd_columns].multiply(dz*10**2, axis='index')  # number / cm^2
    column_sd_df = column_sd_df.iloc[1:]  # drop first row due to NaN in dz
    if column_sd_df.isnull().values.any():
        raise Exception('NaN values prior to integration could cause faulty results across the size distribution')
    column_sd_df = column_sd_df.sum(axis=0)

    return column_sd_df


def get_processed_dust_size_distribution_in_riyadh(date, do_cloud_screening):
    fp = '/Users/osipovs/Data/NCAR/CloudSeeding/Duncan/{}_number.txt'.format(date.strftime('%Y%m%d'))
    # time is the seconds offset for a given date
    df = pd.read_csv(fp, header=0, delim_whitespace=True, parse_dates=['time', ], date_parser=lambda d: date + dt.timedelta(seconds=float(d)))  # , skipinitialspace=True)
    df.rename(columns={'GALT[m]': 'GALT'}, inplace=True)
    df.set_index('time', inplace=True)

    df = df.sort_values('GALT')
    df = df.dropna()
    dz = df.GALT.diff()

    if do_cloud_screening:
        flight_obs_df = get_flight_obs()
        x = (flight_obs_df.index - df.index.min()).total_seconds()
        x_new = (df.index - df.index.min()).total_seconds()
        f = sp.interpolate.interp1d(x, flight_obs_df.SLWC, fill_value="extrapolate")
        df['SLWC'] = f(x_new)

        '''
        Formulate a cloud flag and remove in-cloud data points (PCASP and FSSP data only, DMA data are not thought to be impacted by cloud).
        Threshold for in-cloud is set to FSSP LWC > 0.02 g/m3 or FSSP conc > 25/cm3. In-cloud flag is expanded to assign 'in-cloud' to adjacent points (30 seconds before and after in-cloud data points)
        '''
        df = df[df.SLWC < 0.02]

    # integrate vertically
    sd_columns = []
    bin_sizes = []
    for column in df.columns:
        try:
            size = float(column)
            sd_columns += [column, ]
            bin_sizes += [size, ]
        except:
            continue

    # ASD: aerosol conc in each size bin dN/dlogDp (number/cm^3); aerosol size in micrometers
    # ALT: altitude in meters
    column_sd_df = df[sd_columns].multiply(dz*10**2, axis='index')  # number / cm^2
    column_sd_df = column_sd_df.dropna()
    # column_sd_df = column_sd_df.iloc[1:]  # drop first row due to NaN in dz
    if column_sd_df.isnull().values.any():
        raise Exception('NaN values prior to integration could cause faulty results across the size distribution')
    column_sd_df = column_sd_df.sum(axis=0)

    column_sd_df.index = bin_sizes

    return df, column_sd_df, sd_columns, bin_sizes  # individual and integrated profile


#%% get unprocessed size distribution


def get_flight_obs():
    flight_obs_df = pd.read_csv('/Users/osipovs/Data/NCAR/CloudSeeding/samples/C070409_1.txt', delim_whitespace=True, parse_dates=[[0, 1]])  # delimiter=['\t', ' '])  # the file seems to containt full output
    flight_obs_df.rename(columns={'DATE_TIME':'time'}, inplace=True)
    flight_obs_df.set_index('time', inplace=True)
    return flight_obs_df


def get_probed_flight_obs(date, do_cloud_screening=True):
    df = pd.read_csv('/Users/osipovs/Data/NCAR/CloudSeeding/samples/C{}_1_probes.txt'.format(date.strftime('%y%m%d')), delim_whitespace=True, parse_dates=['TIME', ])  # delimiter=['\t', ' '])  # the file seems to containt full output
    df.rename(columns={'TIME':'time'}, inplace=True)
    df.set_index('time', inplace=True)

    if do_cloud_screening:
        '''
        Formulate a cloud flag and remove in-cloud data points (PCASP and FSSP data only, DMA data are not thought to be impacted by cloud).
        Threshold for in-cloud is set to FSSP LWC > 0.02 g/m3 or FSSP conc > 25/cm3. In-cloud flag is expanded to assign 'in-cloud' to adjacent points (30 seconds before and after in-cloud data points)
        '''
        df = df[df.SLWC < 0.02]

    return df


def get_probed_size_distribution_in_riyadh(date, do_cloud_screening=True):
    df = get_probed_flight_obs(date, do_cloud_screening)

    df = df.dropna()
    df = df.sort_values('GALT')
    dz = df.GALT.diff()

    '''
    get SD subset. Combine PCASP (15 bins from 0.11 to 2.75) and FSSP (20 bins observations from 4 to 45.5)
    '''
    fssp_columns = [column for column in df.columns if 'SPPC_' in column]  # fssp columns
    pcasp_columns = [column for column in df.columns if 'PCASC_' in column]  # fssp columns
    sd_columns = pcasp_columns + fssp_columns

    pcasp_diam_stag = np.linspace(0.11, 3, 16)  # Assumed uniformly spaced 15 bins
    pcasp_diam_rho = (pcasp_diam_stag[1:] + pcasp_diam_stag[:-1]) / 2
    fssp_diam_stag = np.linspace(4, 50, 21)  # Assumed uniformly spaced 20 bins
    fssp_diam_rho = (fssp_diam_stag[1:]+fssp_diam_stag[:-1])/2
    sd_diam_stag = np.concatenate((pcasp_diam_stag, fssp_diam_stag))
    sd_diam_rho = np.concatenate((pcasp_diam_rho, fssp_diam_rho))
    print('TODO:FIND OUT FSSP & PCASP exact BINS')

    column_sd_df = df[sd_columns].multiply(dz * 10 ** 2, axis='index')  # number / cm^2
    column_sd_df = column_sd_df.dropna()  # should drop only first line due to NaN in dz
    if column_sd_df.isnull().values.any():
        raise Exception('NaN values prior to integration could cause faulty results across the size distribution')
    column_sd_df = column_sd_df.sum(axis=0)
    column_sd_df.index = sd_diam_rho

    return df, column_sd_df, sd_columns, sd_diam_stag, sd_diam_rho

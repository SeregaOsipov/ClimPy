import pandas as pd
import numpy as np
import functools
__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

from climpy.utils.file_path_utils import get_root_storage_path_on_hpc

'''
Socio-Economic / Macro parameters utils such UN, World Bank and so on 
for params such as GNI, GDP, population, etc.
'''


def inject_region_info(func):
    '''
    wb stands for World Bank
    '''
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        wb_meta_df = None
        if 'wb_meta_df' in kwargs:
            wb_meta_df = kwargs.pop('wb_meta_df')

        keep_only_classified_regions = True
        if 'keep_only_classified_regions' in kwargs:
            keep_only_classified_regions = kwargs.pop('keep_only_classified_regions')

        rename_countries = True
        if 'rename_countries' in kwargs:
            rename_countries = kwargs.pop('rename_countries')

        df = func(*args, **kwargs)

        if rename_countries:
            # replace country names
            df = df.replace('Egypt, Arab Rep.', 'Egypt')
            df = df.replace('Iran, Islamic Rep.', 'Iran')
            df = df.replace('Syrian Arab Republic', 'Syria')
            df = df.replace('United Arab Emirates', 'UAE')
            df = df.replace('Yemen, Rep.', 'Yemen')
            df = df.replace(to_replace=r'^.*Turkey.*$', value='Turkey', regex=True)  # best way to replace names
            df = df.replace('Turkiye', 'Turkey')

        if wb_meta_df is not None:
            wb_isos = wb_meta_df['iso3'].unique()
            wb_regions = wb_meta_df['region'].unique()

            # countries = df['name'].unique()  # List of unique countries names & iso
            isos = df['iso3'].unique()

            print("Can not determine and assign region for the following countries (they will be ignored): ")
            missing_isos = set(isos).difference(set(wb_isos))  # these isos are not in the World Bank
            for iso in missing_isos:
                country = df[df['iso3'] == iso]['name'].iloc[0]
                print('{}: {}'.format(iso, country))

            isos = set(wb_isos).intersection(set(isos))
            for iso in isos:
                region = wb_meta_df[wb_meta_df['iso3'] == iso]['region']
                df.loc[df['iso3'] == iso, 'region'] = region.iloc[0]

            if keep_only_classified_regions:
                df = df[df['region'].isin(wb_regions)]

        return df
    return wrapper_decorator


@inject_region_info
def prep_gdp():
    file_path = get_root_storage_path_on_hpc() + '/Data/WorldBank/GDP/50b0a69e-6b7f-44bd-be3f-c32b843474b8_Data.csv'
    df = pd.read_csv(file_path, skipfooter=5)#, header=2)  # , usecols=['Variant', 'Location', 'PopTotal', 'Time'])  # , index_col='Time'
    df.rename(columns={'Country Code': 'iso', 'Country Name': 'name'}, inplace=True)

    mapping = {}
    for year in np.arange(1960, 2050):
        mapping['{} [YR{}]'.format(year, year)] = year
    df.rename(columns=mapping, inplace=True)  # update years

    return df


@inject_region_info
def prep_gni():
    file_path = get_root_storage_path_on_hpc() + '/Data/WorldBank/API_NY.GNP.ATLS.CD_DS2_en_csv_v2_3474000.csv'  # World Data # total
    gni_df = pd.read_csv(file_path, header=2)  # , usecols=['Variant', 'Location', 'PopTotal', 'Time'])  # , index_col='Time'
    gni_df = gni_df.iloc[:, :-1]  # drop last column to ignore trailing comma
    gni_df.rename(columns={'Country Code': 'iso3', 'Country Name': 'name'}, inplace=True)

    mapping = {}
    for year in np.arange(1960, 2050):
        mapping['{}'.format(year)] = year
    gni_df.rename(columns=mapping, inplace=True)  # update years

    return gni_df


@inject_region_info
def prep_gni_pre_capita():
    file_path = get_root_storage_path_on_hpc() + '/Data/WorldBank/API_NY.GNP.PCAP.CD_DS2_en_csv_v2_3470973.csv'  # World Data # per capita
    gni_per_capita_df = pd.read_csv(file_path, header=2)  # , usecols=['Variant', 'Location', 'PopTotal', 'Time'])  # , index_col='Time'
    gni_per_capita_df = gni_per_capita_df.iloc[:, :-1]  # drop last column to ignore trailing comma
    gni_per_capita_df.rename(columns={'Country Code': 'iso3', 'Country Name': 'name'}, inplace=True)
    return gni_per_capita_df

@inject_region_info
def prep_population():
    file_path = get_root_storage_path_on_hpc() + '/Data/WorldBank/API_SP.POP.TOTL_DS2_en_csv_v2_3469297.csv'
    population_df = pd.read_csv(file_path, header=2)  # , usecols=['Variant', 'Location', 'PopTotal', 'Time'])  # , index_col='Time'
    population_df = population_df.iloc[:, :-1]  # drop last column to ignore trailing comma
    population_df.rename(columns={'Country Code': 'iso3', 'Country Name': 'name'}, inplace=True)

    mapping = {}
    for year in np.arange(1960, 2050):
        mapping['{}'.format(year)] = year
    population_df.rename(columns=mapping, inplace=True)  # update years
    # derive gni per capita
    # gni_per_capita_derived_df = gni_df.iloc[:, 4:].div(population_df.iloc[:, 4:])
    # gni_per_capita_derived_df['iso'] = gni_df['iso']
    # gni_per_capita_derived_df['name'] = gni_df['name']
    # gni_percapita_derived_df['region'] = gni_df['region']

    return population_df


def prep_world_bank_meta_data():
    # %% Define the 8 regions following the World Bank classification
    file_path = get_root_storage_path_on_hpc() + '/Data/WorldBank/Metadata_Country_API_NY.GNP.PCAP.CD_DS2_en_csv_v2_3470973.csv'  # World Data
    meta_df = pd.read_csv(file_path, header=0)  # , usecols=['Variant', 'Location', 'PopTotal', 'Time'])  # , index_col='Time'
    meta_df = meta_df.iloc[:, :-1]  # drop last column
    meta_df.rename(columns={'Country Code': 'iso3', 'TableName': 'name', 'Region': 'region'}, inplace=True)
    # wb_regions = meta_df['region'].unique()
    # wb_regions = np.delete(wb_regions, 1, 0)  # drop na, should be 7 regions in total
    wb_regions = list(meta_df['region'].unique())
    wb_regions.remove(np.nan)

    meta_df = meta_df[meta_df.region.isin(wb_regions)]  # drop the NaN regions from the meta_df
    wb_isos = meta_df['iso3'].unique()

    return meta_df, wb_regions, wb_isos


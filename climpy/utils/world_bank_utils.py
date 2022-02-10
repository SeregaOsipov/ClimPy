import pandas as pd
import numpy as np
__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

'''
Socio-Economic / Macro parameters utils such UN, World Bank and so on 
for params such as GNI, GDP, population, etc.
'''


def pred_world_bank_data():
    # file_path = '/work/mm0062/b302074/Data/AirQuality/EMME/projection/GNI per Capita UN data.csv'
    file_path = '/work/mm0062/b302074/Data/AirQuality/EMME/projection/WorldBank/API_NY.GNP.PCAP.CD_DS2_en_csv_v2_3470973.csv'  # World Data # per capita
    gni_per_capita_df = pd.read_csv(file_path, header=2)  # , usecols=['Variant', 'Location', 'PopTotal', 'Time'])  # , index_col='Time'
    gni_per_capita_df = gni_per_capita_df.iloc[:, :-1]  # drop last column to ignore trailing comma
    gni_per_capita_df.rename(columns={'Country Code': 'iso', 'Country Name': 'name'}, inplace=True)

    file_path = '/work/mm0062/b302074/Data/AirQuality/EMME/projection/WorldBank/API_NY.GNP.ATLS.CD_DS2_en_csv_v2_3474000.csv'  # World Data # total
    gni_df = pd.read_csv(file_path, header=2)  # , usecols=['Variant', 'Location', 'PopTotal', 'Time'])  # , index_col='Time'
    gni_df = gni_df.iloc[:, :-1]  # drop last column to ignore trailing comma
    gni_df.rename(columns={'Country Code': 'iso', 'Country Name': 'name'}, inplace=True)
    mapping = {}
    for year in np.arange(1960, 2050):
        mapping['{}'.format(year)] = year
    gni_df.rename(columns=mapping, inplace=True)  # update years

    file_path = '/work/mm0062/b302074/Data/AirQuality/EMME/projection/WorldBank/API_SP.POP.TOTL_DS2_en_csv_v2_3469297.csv'
    population_df = pd.read_csv(file_path, header=2)  # , usecols=['Variant', 'Location', 'PopTotal', 'Time'])  # , index_col='Time'
    population_df = population_df.iloc[:, :-1]  # drop last column to ignore trailing comma
    population_df.rename(columns={'Country Code': 'iso', 'Country Name': 'name'}, inplace=True)
    population_df.rename(columns=mapping, inplace=True)  # update years
    # derive gni per capita
    # gni_per_capita_derived_df = gni_df.iloc[:, 4:].div(population_df.iloc[:, 4:])
    # gni_per_capita_derived_df['iso'] = gni_df['iso']
    # gni_per_capita_derived_df['name'] = gni_df['name']
    # gni_percapita_derived_df['region'] = gni_df['region']

    return gni_df, gni_per_capita_df, population_df


def prep_world_bank_meta_data():
    # %% Define the 8 regions following the World Bank classification
    file_path = '/work/mm0062/b302074/Data/AirQuality/EMME/projection/WorldBank/Metadata_Country_API_NY.GNP.PCAP.CD_DS2_en_csv_v2_3470973.csv'  # World Data
    meta_df = pd.read_csv(file_path, header=0)  # , usecols=['Variant', 'Location', 'PopTotal', 'Time'])  # , index_col='Time'
    meta_df = meta_df.iloc[:, :-1]  # drop last column
    meta_df.rename(columns={'Country Code': 'iso', 'TableName': 'name', 'Region': 'region'}, inplace=True)
    wb_regions = meta_df['region'].unique()
    wb_regions = np.delete(wb_regions, 1, 0)  # drop na, should be 7 regions in total
    wb_isos = meta_df['iso'].unique()

    return meta_df, wb_regions, wb_isos


def inject_region_info(df, wb_meta_df, wb_isos):
    '''
    wb stands for World Bank
    '''
    countries = df['name'].unique()  # List of unique countries that we work with
    isos = df['iso'].unique()

    print("Following countries can not be assigned region and will be ignored")
    missing_isos = set(isos).difference(set(wb_isos))  # these isos are not in the World Bank
    for iso in missing_isos:
        country = df[df['iso'] == iso]['name'].iloc[0]
        print('{}: {}'.format(iso, country))

    isos = set(wb_isos).intersection(set(isos))
    for iso in isos:
        region = wb_meta_df[wb_meta_df['iso'] == iso]['region']
        df.loc[df['iso'] == iso, 'region'] = region.iloc[0]

    # return df
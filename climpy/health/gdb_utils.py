from climpy.utils.file_path_utils import get_root_storage_path_on_hpc
import pandas as pd


def prep_gdb_death_rate(metric):
    '''
    prep all cause mortality from GDB for 2017 https://ghdx.healthdata.org/gbd-results-tool?params=gbd-api-2019-permalink/763ac8938ca427297d848478ddac47ea
    metric: 1 is number of deaths, 3 is death rate
    '''
    file_path = get_root_storage_path_on_hpc() + 'Data/AirQuality/AQABA/death_rates/gbd/IHME-GBD_2019_DATA-9a49e703-1.csv'
    death_rate_df = pd.read_csv(file_path)  # deaths and rate
    death_rate_df = death_rate_df[death_rate_df['metric']==metric]
    death_rate_df = death_rate_df[['location', 'val', 'upper', 'lower']]

    # replace codes with names
    fp = '/work/mm0062/b302074/Data/AirQuality/AQABA/death_rates/gbd/IHME_GBD_2019_CODEBOOK/IHME_GBD_2019_CODEBOOK_Y2022M04D06.CSV'
    gbd_meta_df = pd.read_csv(fp, skiprows=range(1, 2))
    gbd_locations_map_df = gbd_meta_df[['location_id', 'location_name']]
    mapping_dict = pd.Series(gbd_locations_map_df.location_name.values, index=gbd_locations_map_df.location_id.values).to_dict()
    mapping_dict[142] = 'Iran, Islamic Republic of'
    mapping_dict[153] = 'Syria'
    death_rate_df = death_rate_df.replace({'location': mapping_dict})
    death_rate_df = death_rate_df.rename(columns={'location': 'country', 'val': 'mean'})
    # drop MENA
    death_rate_df = death_rate_df[death_rate_df['country']!='North Africa and Middle East']

    return death_rate_df


def prep_covid_death_rate():
    # https://www.statista.com/statistics/1104709/coronavirus-deaths-worldwide-per-million-inhabitants/
    file_path = get_root_storage_path_on_hpc() + 'Data/AirQuality/AQABA/death_rates/statistic_id1104709_covid-19-cases-and-deaths-per-million-in-203-countries-as-of-september-3-2021.xlsx'
    corona_df = pd.read_excel(file_path, sheet_name='Data', skiprows=4)

    corona_df = corona_df[['country', 'Deaths per million (total)']]
    corona_df['Deaths per 100.000'] = corona_df['Deaths per million (total)'] / 10  # deaths per 10**6 -> deaths per 100.000
    corona_df['Deaths per 100.000'] *= 12 / 21  # total -> per year, assuming the 17 November 2019 as the beginning date
    corona_df.drop('Deaths per million (total)', axis=1, inplace=True)
    corona_df = corona_df.replace({'country': {'Iran': 'Iran, Islamic Republic of'}})
    return corona_df
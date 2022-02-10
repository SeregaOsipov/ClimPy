import pandas as pd
import numpy as np
import functools
from climpy.utils.world_bank_utils import inject_region_info
__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'


EDGAR_GHG_POLLUTANTS = ['CH4', 'CO2', 'N2O']
EDGAR_AP_POLLUTANTS = ['BC', 'CO', 'NH3', 'NMVOC', 'NOx', 'OC', 'PM10', 'PM2.5', 'SO2']


def to_per_capita(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        population_for_per_capita = None
        if 'population_for_per_capita' in kwargs:
            population_for_per_capita = kwargs.pop('population_for_per_capita')

        df = func(*args, **kwargs)

        if population_for_per_capita is not None:
            print('Convert df to per capita')
            df = df.div(population_for_per_capita)

        return df
    return wrapper_decorator


def aggregate_co2(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):

        pollutant = args[0]

        if pollutant == 'CO2':
            print('CO2 will be aggregated')

            new_args = ('CO2_excl_short-cycle_org_C', ) + args[1:]
            wo_short_df = func(*new_args, **kwargs)
            new_args = ('CO2_org_short-cycle_C',) + args[1:]
            short_df = func(*new_args, **kwargs)

            df = wo_short_df + short_df  # have no idea how it sums
        else:
            df = func(*args, **kwargs)
        return df

    return wrapper_decorator


def aggregate_by_regions(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):

        wb_meta_df = None
        if 'wb_meta_df' in kwargs:
            wb_meta_df = kwargs.pop('wb_meta_df')
        wb_isos = 0
        if 'wb_isos' in kwargs:
            wb_isos = kwargs.pop('wb_isos')
        wb_regions = 0
        if 'wb_regions' in kwargs:
            wb_regions = kwargs.pop('wb_regions')

        df = func(*args, **kwargs)
        print(*args)

        if wb_meta_df is not None:
            inject_region_info(df, wb_meta_df, wb_isos)
            df = df[df['region'].isin(wb_regions)]
            regional_df = df.groupby('region').aggregate(np.sum)

        return regional_df

    return wrapper_decorator


@to_per_capita
@aggregate_co2  # combine short and wo short CO2 variables
@aggregate_by_regions
def prep_edgar_pollutant(pollutant, file_path_template):
    file_path = file_path_template.format(pollutant, pollutant, pollutant)
    edgar_df = pd.read_excel(file_path, sheet_name='TOTALS BY COUNTRY', header=9)
    edgar_df.rename(columns={'ISO_A3': 'iso', 'Country_code_A3': 'iso', 'Name': 'name'}, inplace=True)  # both AP & GHG columns

    mapping = {}
    for year in np.arange(1970, 2019):
        mapping['Y_{}'.format(year)] = year
    edgar_df.rename(columns=mapping, inplace=True)  # update years, GHG only Y_year columns

    return edgar_df


def derive_regional_statistics(regional_df):
    '''
    df has only years-rows and region-columns
    '''
    trend = regional_df.diff()  # per year
    rel_trend = trend.div(regional_df) * 10 ** 2
    growth_since_1970 = (regional_df - regional_df.loc[1970]).div(regional_df.loc[1970]) * 10 ** 2 + 10 ** 2  # growth in percent, start from 100%

    # enum = gni_per_capita.diff().iloc[11:-5]  # rename str to int
    # denum = regional_df.diff().iloc[1:, :]  # drop first NaN
    # sensitivity = enum.div(denum)  # GNI change per emissions change

    return trend, rel_trend, growth_since_1970, None


def prep_edgar_totals_by_region(pollutants, file_path_template, wb_meta_df, wb_isos, wb_regions, population_for_per_capita=None):
    '''
    AP case
    pollutants = ['BC', 'CO', 'NH3', 'NMVOC', 'NOx', 'OC', 'PM10', 'PM2.5', 'SO2']
    file_path_template = '/work/mm0062/b302074/Data/emissions/EDGAR/v50_AP/{}/v50_{}_1970_2015.xls'

    GHG case
    pollutants = ['CH4', 'CO2_excl_short-cycle_org_C', 'CO2_org_short-cycle_C', 'N2O']
    file_path_template = '/work/mm0062/b302074/Data/emissions/EDGAR/v60_GHG/{}/v60_{}_1970_2018.xls'

    wb stands for World Bank
    wb_meta_df holds data how to link countries and regions
    '''

    edgars = {}
    trends = {}
    rel_trends = {}
    growths_since_1970 = {}
    sensitivities = {}
    for pollutant in pollutants:  # pollutant = 'SO2'
        regional_df = prep_edgar_pollutant(pollutant, file_path_template, wb_meta_df=wb_meta_df, wb_isos=wb_isos, wb_regions=wb_regions, population_for_per_capita=population_for_per_capita)
        regional_df = regional_df.transpose()

        edgars[pollutant] = regional_df
        trends[pollutant], rel_trends[pollutant], growths_since_1970[pollutant], sensitivities[pollutant] = derive_regional_statistics(regional_df)

        # trends[pollutant] = edgar_regional_df.transpose().diff()
        # rel_trends[pollutant] = edgar_regional_df.transpose().diff().div(edgar_regional_df.transpose()) * 10 ** 2
        # temp_df = edgar_regional_df.transpose()
        # growths_since_1970[pollutant] = (temp_df-temp_df.loc[1970]).div(temp_df.loc[1970])*10**2 + 10**2  # growth in percent, start from 100%
        #
        # enum = gni_per_capita_regional_df.rename(columns=int).transpose().diff().iloc[11:-5]  # rename str to int
        # denum = edgar_regional_df.transpose().diff().iloc[1:, :]  # drop first NaN
        # sensitivities[pollutant] = enum.div(denum)  # GNI change per emissions change

    return edgars, trends, rel_trends, growths_since_1970, sensitivities



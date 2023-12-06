import pandas as pd

from climpy.utils.file_path_utils import get_root_storage_path_on_hpc
from climpy.utils.world_bank_utils import inject_region_info


@inject_region_info
def get_cams_emissions(drop_ocean=True):
    '''
    CAMS 2050, totals data from Klaus

    Units are Tg yr-1 as in the maps.
    Entries without countries are ocean

    2015 CAMS should be simply ECLIPSE version 6.
    '''
    fp = get_root_storage_path_on_hpc() + '/Data/emissions/CAMS/cams_by_country.csv'  # originally produced by Klaus Klingmuller here /work/mm0062/b302045/emac-daten/cams/cams_by_country.csv
    df = pd.read_csv(fp, header=0)
    df = df.convert_dtypes()
    df['emission'] = df.emission.astype(float)
    df.rename(columns={'iso3': 'iso', 'location_name':'name'}, inplace=True)
    df.drop('country.id', axis=1, inplace=True)
    df = df[df.component != 'co2_excl_short-cycle_org_C']  # the other one is CO2, I assume that it has everything
    df = df[df.sector != 'sum']  # Klaus includes 'sum' in the section column. This leads to double counting, drop it
    if drop_ocean:  # drop rows without iso
        df.dropna(inplace=True)

    # df[df.emission > 10 ** 30] = 0  # np.NaN  # fix the outliers
    print('Replacing erroneous values (mostly voc5 + oc)')
    print('Remember that VOC5 is problematic')
    df.loc[df.emission > 10 ** 30, 'emission'] = 0  # np.NaN  # fix the outliers

    return df

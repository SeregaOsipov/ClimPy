import pandas as pd

SENTINEL_DATA_ROOT_PATH = '/Users/osipovs/Data/Copernicus/Sentinel-5P/'
TROPOMI_in_WRF_KEYS = ['ch4', 'o3', 'hcho', 'so2', 'co', 'no2',]


def get_tropomi_files_metadata(key):
    df = pd.read_csv(SENTINEL_DATA_ROOT_PATH + '/meta/{}_meta.csv'.format(key))
    return df

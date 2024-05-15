import numpy as np
import pandas as pd
import pycountry
import regionmask
import xarray as xr
from climpy.utils.file_path_utils import get_root_storage_path_on_hpc


middle_eastern_countries = 'Algeria, Bahrain, Egypt, Iran, Iraq, Israel, Jordan, Kuwait, Lebanon, Libya, Morocco, Oman, Palestine, Qatar, Saudi Arabia, Syria, Tunisia, Turkey, United Arab Emirates, Yemen'.split(', ')
middle_eastern_countries_in_AQABA_WRF_domain = 'Bahrain, Cyprus, Egypt, Iran, Iraq, Israel, Jordan, Kuwait, Lebanon, Oman, Qatar, Saudi Arabia, Syria, Turkey, United Arab Emirates, Yemen'.split(', ')   #inside WRF domain
middle_eastern_countries_in_AQABA_WRF_domain_for_table = 'Bahrain, Cyprus, Egypt, Iran, Iraq, Israel, Jordan, Kuwait, Lebanon, Oman, Qatar, Saudi Arabia, Syria, Turkey, UAE, Yemen'.split(', ')


def get_present_population():
    # TODO: population does not add up correctly. It is OK as relative weights, but I have to fix the regriding
    ds = xr.open_dataset(get_root_storage_path_on_hpc()+'/Data/AirQuality/EMME/population/GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0_on_wrf_grid.nc').rename({'Band1':'PopulationCount'})
    # test_ds = xr.open_dataset('/Users/osipovs/Data/GHSL/GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0/GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0_MENA.nc')
    return ds


def compute_pop_weighted_diags_by_country(xr_in, population_ds=None):
    xr_in = xr_in.rename({'XLONG': 'lon', 'XLAT': 'lat'})

    # middle_eastern_countries_in_AQABA_WRF_domain.sort()  # sorting happens in place

    # United... -> UAE
    # middle_eastern_countries_in_AQABA_WRF_domain_for_table.sort()
    # middle_eastern_countries_in_AQABA_WRF_domain = middle_eastern_countries_in_AQABA_WRF_domain[0:3]  # debug

    regionmask_countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_50
    mask = regionmask_countries.mask(xr_in.isel(Time=0))  # mask all countries even partial overlap

    def weight_and_average_ds(in_country_ds, population_as_weight_ds=None):
        if population_as_weight_ds is None:
            pop_weighted_in_country_ds = in_country_ds
        else:
            pop_weighted_in_country_ds = in_country_ds.weighted(population_as_weight_ds.PopulationCount.rename({'latitude': 'south_north', 'longitude': 'west_east'}))
        pop_weighted_in_country_ds = pop_weighted_in_country_ds.mean(dim=('south_north', 'west_east'))
        return pop_weighted_in_country_ds

    dss_by_country = []

    # add Middle East in general accounting for all countries in the list first
    country_indices_in_me = regionmask_countries.map_keys(middle_eastern_countries_in_AQABA_WRF_domain)
    in_me_ds = xr_in.where(np.in1d(mask, country_indices_in_me).reshape(mask.shape))
    weighted_ds = weight_and_average_ds(in_me_ds, population_ds)
    dss_by_country += [weighted_ds, ]

    # process countries individually
    for country in middle_eastern_countries_in_AQABA_WRF_domain:
        country_index = regionmask_countries.map_keys(country)
        print('{}:{}'.format(country, country_index))

        in_country_ds = xr_in.where(mask == country_index)
        weighted_ds = weight_and_average_ds(in_country_ds, population_ds)
        dss_by_country += [weighted_ds, ]

    pop_weighted_diags_by_country_ds = xr.concat(dss_by_country, dim='country')
    pop_weighted_diags_by_country_ds['country'] = ['Middle East', ] + middle_eastern_countries_in_AQABA_WRF_domain_for_table  # middle_eastern_countries_in_AQABA_WRF_domain

    return pop_weighted_diags_by_country_ds


def read_csv_from_chowdhury(file_path, diag_key):
    df = pd.read_csv(file_path)

    # drop dummy countries with zeros
    df = df.loc[~(df[diag_key] == 0)]  # m means mean

    names = []  # convert ISO to names
    for iso in df.iso:
        country = pycountry.countries.get(alpha_3=iso)

        name = 'NOT FOUND, ISO:'.format(iso)
        if country is not None:
            name = country.name

        if name == 'Syrian Arab Republic':
            name = 'Syria'
        if name == 'United States':
            name = 'USA'
        names.append(name)
        # print('{}: {}'.format(iso, country.name))

    df['country'] = names

    # adjust names
    df = df.replace('Egypt, Arab Rep.', 'Egypt')
    df = df.replace('Iran, Islamic Rep.', 'Iran')
    df = df.replace('Iran, Islamic Republic of', 'Iran')
    df = df.replace('Syrian Arab Republic', 'Syria')
    df = df.replace('United Arab Emirates', 'UAE')
    df = df.replace('Yemen, Rep.', 'Yemen')

    df.set_index('country', inplace=True)
    df.drop(labels=['label', 'iso'], axis=1, inplace=True)
    if 'm' in df.columns:
        df.rename(columns={'m': 'mean', 'l': 'lower', 'h': 'upper'}, inplace=True)

    return df

import pandas as pd
import xarray as xr
from cartopy import crs as ccrs, feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib import pyplot as plt
import requests
from climpy.utils.file_path_utils import get_root_path_on_hpc, get_root_storage_path_on_hpc
import os

SENTINEL_DATA_ROOT_PATH = get_root_storage_path_on_hpc() + '/Data/Copernicus/Sentinel-5P/'
TROPOMI_in_WRF_KEYS = ['ch4', 'o3', 'hcho', 'so2', 'co', 'no2',]


def configure_tropomi_credentials():
    '''
    Setup Access Keys
    credentials for access: https://eodata-s3keysmanager.dataspace.copernicus.eu/panel/s3-credentials
    '''
    os.environ['CDSE_S3_ACCESS'] = 'WAG1HIA5XM3OHAGB70OX'
    os.environ['CDSE_S3_SECRET'] = 'nOf3MHPhj0EzIP3xpP3nEebHUgJAnXPyCs1HAWRq'


def fetch_online_tropomi_metadata(key):
    '''
    ask for Reprocessing (RPRO) rather than Offline (OFFL)
    Request for L2__SO2___ L2__CH4___ L2__HCHO__ L2__CO____ L2__NO2___ L2__O3____
    :param key:
    :return:
    '''

    tropomi_key = 'L2__{}___'.format(key)

    url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq 'SENTINEL-5P' and ContentDate/Start gt 2023-06-01T00:00:00.000Z and ContentDate/Start lt 2023-06-03T00:00:00.000Z and OData.CSC.Intersects(area=geography'SRID=4326;POLYGON((44 22, 57 22, 57 35, 44 35, 44 22))') and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productClass' and att/OData.CSC.StringAttribute/Value eq 'RPRO') and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'processingLevel' and att/OData.CSC.StringAttribute/Value eq 'L2') and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and (att/OData.CSC.StringAttribute/Value eq '{}'))&$top=100&$expand=Attributes&$orderby=ContentDate/Start".format(tropomi_key)

    url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq 'SENTINEL-5P' and ContentDate/Start gt 2023-06-01T00:00:00.000Z and ContentDate/Start lt 2023-07-01T00:00:00.000Z and OData.CSC.Intersects(area=geography'SRID=4326;POLYGON((44 22, 57 22, 57 35, 44 35, 44 22))') and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and (att/OData.CSC.StringAttribute/Value eq '{}'))&$top=100&$expand=Attributes&$orderby=ContentDate/Start".format(tropomi_key)

    print('Getting a request for {} at:\n{}'.format(key, url))
    json = requests.get(url).json()
    df = pd.DataFrame.from_dict(json['value'])
    # Additional filtering for SO2, to remove RPRO and only keep OFFL cases
    df = df[df['Name'].str.contains('OFFL', case=False, na=False)]

    return df


def get_tropomi_files_metadata(key):
    df = pd.read_csv(SENTINEL_DATA_ROOT_PATH + '/meta/{}_meta.csv'.format(key))

    # derive dates and them as a separate column
    dates = df.Name.apply(derive_dates_from_filename)
    df['start_date'] = dates.apply(lambda x: x[0])
    df['end_date'] = dates.apply(lambda x: x[1])

    return df


def derive_tropomi_ch4_pressure_grid(tropomi_ds):
    with xr.set_options(keep_attrs=True):
        tropomi_ds['p_stag'] = tropomi_ds.surface_pressure - tropomi_ds.level * tropomi_ds.pressure_interval
        tropomi_ds.p_stag.attrs['long_name'] = 'pressure grid'


def derive_tropomi_no2_pressure_grid(tropomi_ds):
    with xr.set_options(keep_attrs=True):
        # a staggered grid depends on dim vertices (low and high pressure)
        tropomi_ds['p_stag'] = tropomi_ds.tm5_constant_a + tropomi_ds.tm5_constant_b*tropomi_ds.surface_pressure
        tropomi_ds.p_stag.attrs['units'] = tropomi_ds.surface_pressure.units
        tropomi_ds.p_stag.attrs['long_name'] = 'staggered pressure grid'
        # derive pressure
        tropomi_ds['p_rho'] = tropomi_ds.p_stag.mean(dim='vertices')
        tropomi_ds.p_rho.attrs['long_name'] = 'rho pressure grid'


def prep_tropomi_data(fp):
    ds = xr.open_dataset(fp, group='PRODUCT')
    ds = ds.set_coords(['latitude', 'longitude'])
    ds = ds.squeeze()

    meta_ds = xr.open_dataset(fp, group='METADATA')
    return ds, meta_ds


def derive_dates_from_filename(file_name):
    start_date = pd.to_datetime(file_name[20:35], format='%Y%m%dT%H%M%S')
    end_date = pd.to_datetime(file_name[36:51], format='%Y%m%dT%H%M%S')
    return start_date, end_date


def derive_information_fraction(df):
    names_df = df.Name

    fractions = []
    for filename in names_df:  # .iloc[0:2]:
        print(filename, end='\r')
        tropomi_key = 'methane_mixing_ratio_bias_corrected'
        fp = SENTINEL_DATA_ROOT_PATH + filename
        ds, meta_ds = prep_tropomi_data(fp)

        # Original DS
        qa_da = ds.qa_value
        da = ds[tropomi_key].where(qa_da > 0.8)

        information_fraction = 1 - da.isnull().sum() / da.size
        fractions.append(information_fraction.item())

    df['Information_fraction'] = fractions


def visualize_pcolormesh(data_array, longitude, latitude, projection, color_scale, unit, long_name, vmin, vmax, set_global=True, lonmin=-180, lonmax=180, latmin=-90, latmax=90, ax=None, fig=None):
    if ax is None:
        fig = plt.figure()  # figsize=(6.4, 4.8))
        ax = plt.axes(projection=projection)

    img = ax.pcolormesh(longitude, latitude, data_array, cmap=plt.get_cmap(color_scale), transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax, shading='auto')

    ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=1)
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1)

    if projection==ccrs.PlateCarree():
        ax.set_extent([lonmin, lonmax, latmin, latmax], projection)
        gl = ax.gridlines(draw_labels=True, linestyle='--')
        gl.top_labels=False
        gl.right_labels=False
        gl.xformatter=LONGITUDE_FORMATTER
        gl.yformatter=LATITUDE_FORMATTER
        gl.xlabel_style={'size':14}
        gl.ylabel_style={'size':14}

    if set_global:
        ax.set_global()
        ax.gridlines()

    cbar = fig.colorbar(img, ax=ax, orientation='horizontal', fraction=0.04, pad=0.1)
    cbar.set_label(unit, fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    ax.set_title(long_name, fontsize=20, pad=20.0)

    return fig, ax

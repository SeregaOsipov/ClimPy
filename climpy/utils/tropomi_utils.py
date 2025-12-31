from types import SimpleNamespace

import pandas as pd
import xarray as xr
from cartopy import crs as ccrs, feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib import pyplot as plt
from climpy.utils.file_path_utils import get_root_path_on_hpc, get_root_storage_path_on_hpc
import os
import glob
import requests
import pandas as pd
from netCDF4 import Dataset
from wrf import geo_bounds
from datetime import datetime
from argparse import Namespace
from examples.regridding.regrid_tropomi_on_wrf_grid import regrid_tropomi_on_wrf_grid
from netCDF4 import Dataset
from wrf import getvar, geo_bounds

SENTINEL_DATA_ROOT_PATH = get_root_storage_path_on_hpc() + '/Data/Copernicus/Sentinel-5P/'
TROPOMI_in_WRF_KEYS = ['ch4', 'o3', 'hcho', 'so2', 'co', 'no2',]

# Define QA mapping for different keys. Default to 0.5 if key is not found, as it's the standard for most species
QA_THRESHOLDS = {
    'nitrogendioxide_tropospheric_column': 0.75,
    'methane_mixing_ratio_bias_corrected': 0.8,
    'formaldehyde_tropospheric_vertical_column': 0.5,
    'sulfurdioxide_total_vertical_column': 0.5,
    'carbonmonoxide_total_vertical_column': 0.5
}


def get_tropomi_configs():
    ch4_settings = SimpleNamespace(diag_key='ch4', tropomi_key='methane_mixing_ratio_bias_corrected')#, wrf_key='xch4_like_tropomi')
    no2_settings = SimpleNamespace(diag_key='no2', tropomi_key='nitrogendioxide_tropospheric_column')#, wrf_key='trop_no2_column_like_tropomi')
    return ch4_settings, no2_settings


def get_wrf_polygon(wrf_file_path):
    """Extracts the domain boundary from a wrfout file as a WKT Polygon."""
    # wrf-python returns a GeoBounds object with bottom_left and top_right
    bounds = geo_bounds(wrfin=Dataset(wrf_file_path))

    west = bounds.bottom_left.lon
    south = bounds.bottom_left.lat
    east = bounds.top_right.lon
    north = bounds.top_right.lat

    # Format as a closed POLYGON for the Copernicus OData API
    # Order: (Lon Lat, Lon Lat, ...)
    wkt_polygon = (f"POLYGON(({west} {south}, {east} {south}, "
                   f"{east} {north}, {west} {north}, {west} {south}))")
    return wkt_polygon


def configure_tropomi_credentials():
    '''
    Setup Access Keys
    credentials for access: https://eodata-s3keysmanager.dataspace.copernicus.eu/panel/s3-credentials
    '''
    os.environ['CDSE_S3_ACCESS'] = 'WAG1HIA5XM3OHAGB70OX'
    os.environ['CDSE_S3_SECRET'] = 'nOf3MHPhj0EzIP3xpP3nEebHUgJAnXPyCs1HAWRq'


def fetch_tropomi_from_wrf_folder(key, folder_path, wrf_date_format='%Y-%m-%d_%H_%M_%S'):
    '''
    The goal: Given the folder with WRF output, get the list of overlapping TROPOMI files
    '''

    # 1. Get list of wrfout files
    fps = sorted(glob.glob(os.path.join(folder_path, "wrfout_d*")))
    if not fps:
        print("No wrfout files found in the directory.")
        return None

    # 2. Determine Spatial Bounds (using the first file)
    nc_sample = Dataset(fps[0])
    bounds = geo_bounds(wrfin=nc_sample)

    west, south = bounds.bottom_left.lon, bounds.bottom_left.lat
    east, north = bounds.top_right.lon, bounds.top_right.lat

    wkt_polygon = (f"POLYGON(({west} {south}, {east} {south}, "
                   f"{east} {north}, {west} {north}, {west} {south}))")

    wrf_dates = sorted([datetime.strptime(f[-19:], wrf_date_format) for f in fps])
    start_date = min(wrf_dates).strftime('%Y-%m-%d')
    end_date = max(wrf_dates).strftime('%Y-%m-%d')

    # 4. Construct OData Query
    tropomi_key = f'L2__{key.upper()}___'
    base_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

    query_filter = (
        f"$filter=Collection/Name eq 'SENTINEL-5P' "
        f"and ContentDate/Start gt {start_date}T00:00:00.000Z "
        f"and ContentDate/Start lt {end_date}T23:59:59.000Z "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{wkt_polygon}') "
        f"and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
        f"and (att/OData.CSC.StringAttribute/Value eq '{tropomi_key}'))"
    )

    full_url = f"{base_url}?{query_filter}&$top=1000&$expand=Attributes&$orderby=ContentDate/Start"

    # 5. Execute Request
    print(f"Searching {key} from {start_date} to {end_date}")
    print(f"Bounds: {west}, {south} to {east}, {north}")

    response = requests.get(full_url)
    if response.status_code != 200:
        print(f"API Error: {response.status_code}")
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(response.json()['value'])

    # Filter for Offline (OFFL) as per your original requirement
    if not df.empty:
        df = df[df['Name'].str.contains('OFFL', case=False, na=False)]

    add_dates_to_metadata(df)

    return df


# Usage:
# df_metadata = fetch_tropomi_from_wrf_folder('NO2', '/path/to/wrf/output/')


def fetch_online_tropomi_metadata(key):  # TBD
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


def add_dates_to_metadata(df):
    # derive dates and them as a separate column
    dates = df.Name.apply(derive_dates_from_filename)
    df['start_date'] = dates.apply(lambda x: x[0])
    df['end_date'] = dates.apply(lambda x: x[1])


def get_tropomi_files_metadata(fp):
    df = pd.read_csv(fp)
    add_dates_to_metadata(df)
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


def derive_information_fraction(meta_df, tropomi_key, wrf_grid_id=None):
    names_df = meta_df.Name

    # Get the specific threshold or use 0.5 as a safe fallback
    current_qa_limit = QA_THRESHOLDS.get(tropomi_key, 0.5)

    fractions = []
    for filename in names_df:  # .iloc[0:2]:
        print(filename, end='\r')
        fp = SENTINEL_DATA_ROOT_PATH + '/{}/{}'.format(wrf_grid_id, filename)  # Original or regridded TROPOMI onto WRF grid
        # ds, meta_ds = prep_tropomi_data(fp)  # regridded products do not use the groups in netcdf file
        ds = xr.open_dataset(fp)

        # Original DS
        qa_da = ds.qa_value
        da = ds[tropomi_key].where(qa_da > current_qa_limit)

        information_fraction = 1 - da.isnull().sum() / da.size
        fractions.append(information_fraction.item())

    meta_df['Information_fraction'] = fractions


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


def regrid_tropomi_on_wrf_grid_in_batch(tropomi_meta_df: pd.DataFrame, wrf_grid: str = 'AQABA_d01', tropomi_key: str = 'methane_mixing_ratio_bias_corrected'):
    '''

    :param tropomi_meta_df: csv saved via fetch_tropomi_from_wrf_folder
    :param wrf_grid: Unique grid ID. Also a folder in SENTINEL_DATA_ROOT_PATH. Link geo_em file to uniquely identify WRF grid
    :param regridded_tropomi_folder_path:
    :param tropomi_key:
    :return:
    '''

    # Get Path to the regridding script
    climpy_dir = os.environ.get('ClimPy')
    if climpy_dir:
        script_path = os.path.join(climpy_dir, 'regrid_tropomi_on_wrf_grid.py')
        print(f"Full Path: {script_path}")
    else:
        script_path = '/home/osipovs/PycharmProjects/ClimPy/examples/regridding/regrid_tropomi_on_wrf_grid.py'
        print("Error: Environment variable 'ClimPy' is not set. Assume Workstation {}".format(script_path))

    # regridding args
    regridded_tropomi_folder_path = SENTINEL_DATA_ROOT_PATH + '/{}/'.format(wrf_grid)
    os.makedirs(regridded_tropomi_folder_path, exist_ok=True)

    names_ps = tropomi_meta_df.Name
    for index, name in names_ps.items():
        print('\n{}, Processing {}'.format(index, name))
        wrf_in = regridded_tropomi_folder_path + '/geo_em.nc'
        tropomi_in = SENTINEL_DATA_ROOT_PATH + '/{}'.format(name)
        tropomi_out = regridded_tropomi_folder_path + '/{}'.format(name)

        if os.path.exists(tropomi_out):
            print(f"  -> Skipping: Output already exists at {tropomi_out}")
            continue

        # Jupyter Notebook version
        # %run $script_path - -wrf_in = {wrf_in} - -tropomi_in = {tropomi_in} - -tropomi_out = {tropomi_out} - -tropomi_key = {tropomi_key}

        # Pure python version
        args = Namespace(
            wrf_in=wrf_in,
            tropomi_in=tropomi_in,
            tropomi_out=tropomi_out,
            tropomi_key=tropomi_key
        )
        regrid_tropomi_on_wrf_grid(args)
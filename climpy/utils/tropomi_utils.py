import glob
import os
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import requests
import xarray as xr
from cartopy import crs as ccrs, feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib import pyplot as plt
from matplotlib import ticker
from netCDF4 import Dataset
from pystac_client import Client
from wrf import geo_bounds

from climpy.utils.file_path_utils import get_root_storage_path_on_hpc
from examples.regridding.regrid_tropomi_on_wrf_grid import regrid_tropomi_on_wrf_grid

SENTINEL_DATA_ROOT_PATH = get_root_storage_path_on_hpc() + '/Data/Copernicus/Sentinel-5P/'
TROPOMI_in_WRF_KEYS = ['ch4', 'o3', 'hcho', 'chocho', 'so2', 'co', 'no2', 'o3_pr']

# Define QA mapping for different keys. Default to 0.5 if key is not found, as it's the standard for most species
QA_THRESHOLDS = {
    'nitrogendioxide_tropospheric_column': 0.75,
    'methane_mixing_ratio_bias_corrected': 0.8,
    'formaldehyde_tropospheric_vertical_column': 0.5,
    'sulfurdioxide_total_vertical_column': 0.75,  # 0.75 is stricter, otherwise 0.5
    'carbonmonoxide_total_column_corrected': 0.5,
    'ozone_profile': 0.5
}


def get_tropomi_species_configs():
    '''
    cdse_tropomi_key: used in online request to CDSE
    tropomi_key: variable name in the TROPOMI netcdf file
    :return:
    '''
    ch4_settings = SimpleNamespace(diag_key='ch4', tropomi_key='methane_mixing_ratio_bias_corrected', cdse_tropomi_key='L2__CH4___')
    no2_settings = SimpleNamespace(diag_key='no2', tropomi_key='nitrogendioxide_tropospheric_column', cdse_tropomi_key='L2__NO2___')
    so2_settings = SimpleNamespace(diag_key='so2', tropomi_key='sulfurdioxide_total_vertical_column', cdse_tropomi_key='L2__SO2___')
    # o3_settings = SimpleNamespace(diag_key='o3', tropomi_key='ozone_profile', cdse_tropomi_key='L2__O3__PR')
    co_settings = SimpleNamespace(diag_key='co', tropomi_key='carbonmonoxide_total_column_corrected', cdse_tropomi_key='L2__CO____')
    hcho_settings = SimpleNamespace(diag_key='hcho', tropomi_key='formaldehyde_tropospheric_vertical_column', cdse_tropomi_key='L2__HCHO__')
    chocho_settings = SimpleNamespace(diag_key='chocho', tropomi_key='glyoxal_tropospheric_vertical_column', cdse_tropomi_key='L2__CHOCHO__')
    # return [ch4_settings, no2_settings, so2_settings, o3_settings, co_settings, hcho_settings]
    return [ch4_settings, no2_settings, so2_settings, co_settings, hcho_settings, chocho_settings]


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


def fetch_tropomi_metadata_online(start_date, end_date, wkt_polygon, cdse_tropomi_key):
    # 4. Construct OData Query
    base_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

    query_filter = (
        f"$filter=Collection/Name eq 'SENTINEL-5P' "
        f"and ContentDate/Start gt {start_date}T00:00:00.000Z "
        f"and ContentDate/Start lt {end_date}T23:59:59.000Z "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{wkt_polygon}') "
        f"and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
        f"and (att/OData.CSC.StringAttribute/Value eq '{cdse_tropomi_key}'))"
    )

    full_url = f"{base_url}?{query_filter}&$top=1000&$expand=Attributes&$orderby=ContentDate/Start"
    # print(full_url)

    # 5. Execute Request
    response = requests.get(full_url)
    if response.status_code != 200:
        print(f"API Error: {response.status_code}")
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(response.json()['value'])

    # Filter for Offline (OFFL) as per your original requirement
    if not df.empty:
        df = df[df['Name'].str.contains('OFFL', case=False, na=False)]

    return df


def fetch_tropomi_chocho_metadata_online(start_time, end_time, bbox, download_dir=None):
    """
    Queries the S5P-PAL STAC API for CHOCHO orbits.
    Defaults perfectly match the THOFA CDSE OData query:
    Time: June 2023
    Bounding Box: [min_lon, min_lat, max_lon, max_lat] = [44, 22, 57, 35]
    """
    # Debug
    # start_time = "2023-06-01T00:00:00Z"
    # end_time = "2023-07-01T00:00:00Z"
    # bbox = [44.0, 22.0, 57.0, 35.0]

    if download_dir is None:
        download_dir = SENTINEL_DATA_ROOT_PATH

    # Connect to the S5P-PAL STAC Catalog
    catalog_url = "https://data-portal.s5p-pal.com/api/s5p-l2"
    client = Client.open(catalog_url)

    # Format datetime exactly as requested by the STAC API (Start/End)
    time_filter = f"{start_time}/{end_time}"

    print(f"Searching S5P-PAL for l2__chocho between {time_filter} in bbox {bbox}...")

    search = client.search(
        datetime=time_filter,
        bbox=bbox,
        collections=['L2__CHOCHO'],
        limit=100  # Matches your &$top=100 filter
    )

    items = list(search.items())
    meta_records = []
    for item in items:
        # --- OFFICIAL S5P-PAL ASSET PARSING ---
        product = item.assets["product"]
        extra_fields = product.extra_fields

        download_url = product.href
        safe_filename = extra_fields["file:local_path"]
        expected_size = extra_fields["file:size"]
        # --------------------------------------

        filepath = os.path.join(download_dir, safe_filename)

        meta_records.append({
            'Name': safe_filename,
            'safe_filename': safe_filename,
            'download_url': download_url,
            'tropomi_granule_fp': filepath,
            'expected_size': expected_size
        })

    meta_df = pd.DataFrame(meta_records)

    if not meta_df.empty:
        print(f"Successfully loaded {len(meta_df)} CHOCHO orbits.")
    else:
        print("No CHOCHO orbits found for this query.")

    return meta_df


def fetch_tropomi_from_wrf_folder(config, wrf_date_format='%Y-%m-%d_%H_%M_%S'):
    '''
    The goal: Given the folder with WRF output, get the list of overlapping TROPOMI files
    '''

    # 1. Get list of wrfout files
    fps = sorted(glob.glob(os.path.join(config.wrf_output_folder_path, "wrfout_d*")))
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

    print(f"Searching {config.cdse_tropomi_key} from {start_date} to {end_date}")
    print(f"Bounds: {west}, {south} to {east}, {north}")

    if config.diag_key == 'chocho':
        bbox = [west, south, east, north]
        df = fetch_tropomi_chocho_metadata_online(start_date, end_date, bbox)  # config.download_dir)
    else:
        df = fetch_tropomi_metadata_online(start_date, end_date, wkt_polygon, config.cdse_tropomi_key)

    add_dates_to_metadata(df)

    return df


def fetch_online_tropomi_metadata_TBD(key):  # TBD
    '''
    TODO:TBD

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


def prepare_tropomi_meta_data(config):
    print(config.tropomi_meta_data_fps[config.diag_key])
    # initialize_config(config)
    meta_df = get_tropomi_files_metadata(config.tropomi_meta_data_fps[config.diag_key])
    print('meta size before filtering: {}'.format(meta_df.index.size))

    # Only keep orbits with enough data
    derive_information_fraction(meta_df, config.tropomi_key, config.wrf_grid_id)
    meta_df = meta_df[(meta_df['Information_fraction'] > 0.15)]
    print('meta size after filtering: {}'.format(meta_df.index.size))
    return meta_df


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


def derive_tropomi_so2_pressure_grid(tropomi_ds):
    '''
    Vertical Grid: https://sentiwiki.copernicus.eu/__attachments/1673595/S5P-L2-DLR-PUM-400E%20-%20Sentinel-5P%20Level%202%20Product%20User%20Manual%20Sulphur%20Dioxide%20SO2%202024%20-%202.8.0.pdf#page=19.52
    :param tropomi_ds:
    :return:
    '''
    with xr.set_options(keep_attrs=True):
        tropomi_ds['p_rho'] = tropomi_ds.tm5_constant_a + tropomi_ds.tm5_constant_b * tropomi_ds.surface_pressure
        tropomi_ds.p_rho.attrs['units'] = tropomi_ds.surface_pressure.units
        tropomi_ds.p_rho.attrs['long_name'] = 'rho pressure grid'


def derive_tropomi_o3_pr_pressure_grid(tropomi_ds):
    with xr.set_options(keep_attrs=True):
        tropomi_ds['p_rho'] = tropomi_ds.pressure
        tropomi_ds.p_rho.attrs['long_name'] = 'rho pressure grid'


def derive_tropomi_co_pressure_grid(tropomi_ds):
    '''
    pressure_levels: Pressure of the layer interfaces of the vertical grid. The pressures indicate the pressure at the bottom of each layer. The topmost layer extends to the top of atmosphere.
    :param tropomi_ds:
    :return:
    '''
    with xr.set_options(keep_attrs=True):
        # for CO product, layer means rho grid. But pressure is given at bottom interface and reuses same dimension (but should be level or staggered). Here I'm fixing it.
        toa_pressure = xr.zeros_like(tropomi_ds.pressure_levels.isel(layer=0))
        toa_pressure = toa_pressure.expand_dims('layer', axis=-1)
        p_stag_grid = xr.concat([toa_pressure, tropomi_ds.pressure_levels], dim='layer')
        p_stag_grid = p_stag_grid.rename({'layer': 'level'})
        tropomi_ds['p_stag'] = p_stag_grid
        tropomi_ds.p_stag.attrs['long_name'] = 'staggered pressure grid'

        tropomi_ds['p_rho'] = tropomi_ds.p_stag.rolling(level=2).mean().isel(level=slice(1, None)).rename({'level': 'layer'})
        tropomi_ds.p_rho.attrs['long_name'] = 'rho pressure grid'


def derive_tropomi_hcho_pressure_grid(tropomi_ds):
    '''
    Vertical Grid: Same as NO2/SO2 (TM5 based)
    Section 8.5: https://sentiwiki.copernicus.eu/__attachments/1673595/S5P-L2-DLR-PUM-400F%20-%20Sentinel-5P%20Level%202%20Product%20User%20Manual%20Formaldehyde%20HCHO%202022%20-%202.4.pdf?inst-v=87ef0ca0-8091-4ed6-bc9f-05ea3a6bc632#page=17.75
    :param tropomi_ds:
    :return:
    '''
    derive_tropomi_so2_pressure_grid(tropomi_ds)


def derive_tropomi_chocho_pressure_grid(tropomi_ds):
    tropomi_ds['p_rho'] = tropomi_ds.glyoxal_profile_apriori_pressure
    # tropomi_ds.p_rho.attrs['units'] = tropomi_ds.surface_pressure.units
    # tropomi_ds.p_rho.attrs['long_name'] = 'rho pressure grid'

def prep_tropomi_data(fp):
    ds = xr.open_dataset(fp, group='PRODUCT')
    if 'latitude' in ds.coords or 'latitude' in ds.data_vars:
        ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    ds = ds.set_coords(['lat', 'lon'])
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

    # Format the ticks to use scientific notation
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))  # Triggers notation if < 0.001 or > 1000
    cbar.ax.xaxis.set_major_formatter(formatter)

    cbar.set_label(unit, fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.xaxis.get_offset_text().set_fontsize(16)  # # 2. Increase the size of the multiplier (e.g., the 10^-6)

    ax.set_title(long_name, fontsize=20, pad=20.0)

    return fig, ax, cbar


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

        # print(f'--wrf_in={wrf_in} --tropomi_in={tropomi_in} --tropomi_out={tropomi_out} --tropomi_key={tropomi_key}')

        # Pure python version
        args = Namespace(
            wrf_in=wrf_in,
            tropomi_in=tropomi_in,
            tropomi_out=tropomi_out,
            tropomi_key=tropomi_key
        )
        regrid_tropomi_on_wrf_grid(args)


def download_tropomi(meta_df):
    '''
    Getting List of TROPOMI files, CDSE Approach
    Alternative way to Download TROPOMI: https://gist.github.com/nicholasbalasus/008b34590fbc55b757fbd879bb64ccc0
    :param meta_df:
    :return:
    '''
    import eofetch

    sentinel_data_root_path = SENTINEL_DATA_ROOT_PATH
    display('Downloading TROPOMI files to: {}'.format(sentinel_data_root_path))
    configure_tropomi_credentials()

    for index, row in meta_df.iterrows():
        print('downloading {}'.format(row.Name), end='\r')
        eofetch.download(row.Name, target_directory=sentinel_data_root_path)

    print("Done")


def download_tropomi_chocho(meta_df):
    """
    Step 2: Takes the metadata DataFrame and downloads the missing L2 orbit files.
    """

    if meta_df.empty:
        print("Metadata DataFrame is empty. Nothing to download.")
        return meta_df

    print(f"Checking and downloading {len(meta_df)} CHOCHO files...")

    # Using itertuples to match your existing tropomi_utils loop style
    for row in meta_df.itertuples():
        filepath = row.tropomi_granule_fp
        download_url = row.download_url
        safe_filename = row.safe_filename

        # Ensure the target directory exists before writing
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if os.path.exists(filepath):
            print(f"  -> Skipping: {safe_filename} (already downloaded)")
            continue

        print(f"  -> Downloading: {safe_filename}...")
        try:
            # Stream the large NetCDF file in chunks to save RAM
            with requests.get(download_url, stream=True) as r:
                r.raise_for_status()
                with open(filepath, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        except Exception as e:
            print(f"     Failed to download {safe_filename}: {e}")

    print("Download sequence complete.")
    return meta_df


def prepare_tropomi_metadata(config):
    diag_fp = config.tropomi_meta_data_fps[config.diag_key]
    if os.path.exists(diag_fp):
        print('Reading existing metadata from {}'.format(diag_fp))
        meta_df = pd.read_csv(diag_fp)
    else:
        meta_df = fetch_tropomi_from_wrf_folder(config)#, wrf_date_format='%Y-%m-%d_%H:%M:%S')

        if config.wrf_filter_dates: meta_df = meta_df[meta_df['start_date'].between(config.wrf_filter_dates[0], config.wrf_filter_dates[1])]

        # Build the WRF post-processing list to derive TROPOMI-like diagnostics
        meta_df['wrfin_file'] = meta_df['start_date'].apply(lambda x: x.strftime('wrfout_d01_%Y-%m-%d_00_00_00'))  # name of the wrf output
        meta_df['wrfout_file'] = meta_df['start_date'].apply(lambda x: x.strftime('wrfout_d01_%Y-%m-%d_%H_%M_%S'))  # name for the pp-ed wrf file

        # Save list of files for processing
        print('Saving metadata to \n{}'.format(diag_fp))
        Path(diag_fp).parent.mkdir(parents=True, exist_ok=True)  # # 2. Extract the parent directory and create it
        meta_df.to_csv(diag_fp, index=False)#, header=['Name'])

    if 'Id' in meta_df.columns:
        display(meta_df[['Id', 'Name', 'S3Path', 'GeoFootprint']].head(3))
        display(meta_df[['Name', 'start_date', 'wrfin_file' , 'wrfout_file']].head(3))

    return meta_df


def process_metadata_deriving_tropomi_like_diags(meta_df, config):
    meta_df['wrf_tropomi_like_diag_fp'] = meta_df.wrfout_file.apply(lambda x: Path(config.wrf_output_folder_path) / 'pp/tropomi_like_{}/'.format(config.diag_key.lower()) / x)  # add file name of the wrf output sampled at station locations
    meta_df['wrfout_fp'] = meta_df.wrfin_file.apply(lambda x: Path(config.wrf_output_folder_path) / x)  # add file name of the wrf output sampled at station locations
    meta_df['tropomi_granule_fp'] = meta_df.Name.apply(lambda x: Path(SENTINEL_DATA_ROOT_PATH) / '{}'.format(config.wrf_grid_id) / x)

    for row in meta_df.itertuples():
        wrf_in = row.wrfout_fp
        wrf_out = row.wrf_tropomi_like_diag_fp
        tropomi_in = row.tropomi_granule_fp

        if os.path.exists(wrf_out):
            print(f"  -> Skipping: Output already exists at {wrf_out}")
            continue

        # %run /home/osipovs/PycharmProjects/ClimPy/climpy/wrf/pp_wrf_like_tropomi_ch4.py --wrf_in={wrf_in} --wrf_out={wrf_out} --tropomi_in={tropomi_in}

        # Pure python version
        args = Namespace(
            wrf_in=wrf_in,
            wrf_out=wrf_out,
            tropomi_in=tropomi_in,
        )

        from climpy.wrf.pp_wrf_like_tropomi_ch4 import pp_wrf_like_tropomi_ch4
        from climpy.wrf.pp_wrf_like_tropomi_co import pp_wrf_like_tropomi_co
        from climpy.wrf.pp_wrf_like_tropomi_hcho import pp_wrf_like_tropomi_hcho
        from climpy.wrf.pp_wrf_like_tropomi_no2 import pp_wrf_like_tropomi_no2
        from climpy.wrf.pp_wrf_like_tropomi_so2 import pp_wrf_like_tropomi_so2
        from climpy.wrf.pp_wrf_like_tropomi_chocho import pp_wrf_like_tropomi_chocho

        processors = {
            'ch4': pp_wrf_like_tropomi_ch4,
            'no2': pp_wrf_like_tropomi_no2,
            'so2': pp_wrf_like_tropomi_so2,
            'co' : pp_wrf_like_tropomi_co,
            'hcho': pp_wrf_like_tropomi_hcho,
            'chocho': pp_wrf_like_tropomi_chocho,
        }
        pp_wrf_like_tropomi_impl = processors[config.diag_key.lower()]
        pp_wrf_like_tropomi_impl(args)

        # print('\nNext item in meta_df\n')

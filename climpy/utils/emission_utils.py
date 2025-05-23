import xarray as xr
import pandas as pd
import xoak
from climpy.utils.wrf_utils import generate_xarray_uniform_time_data


def prep_wrf_emissions(fp):
    # fp = '/work/mm0062/b302074/Data/AirQuality/THOFA/emissions/HERMESv3_radm2_madesorgam_20230515_THOFA_EDGARv61_HTAPv3_voc_CAMS_ship_OMI_so2.nc'
    # fp = '/Users/osipovs/Data/AirQuality/THOFA/emissions/HERMESv3_radm2_madesorgam_20230515_THOFA_EDGARv61_HTAPv3_voc_CAMS_ship_OMI_so2.nc'
    emissions_ds = xr.open_dataset(fp)
    emissions_ds = emissions_ds.isel(emissions_zdim=0)

    def preprocess_ds(ds):
        time_strs = generate_xarray_uniform_time_data(ds.Times)
        time_datetime = pd.to_datetime(time_strs)
        ds = ds.assign(Time=time_datetime)
        if 'Times' in ds.data_vars.keys():
            ds = ds.drop_vars('Times')

        ds = ds.rename({'Time':'time'})
        return ds

    emissions_ds = preprocess_ds(emissions_ds)
    # emissions_ds = emissions_ds.rename({'south_north':'latitude', 'west_east':'longitude'})

    return emissions_ds


def prep_wrf_emissions_injecting_geo_em_coordinates(emissions_fp, geo_em_fp):
    # Suuplements the missing the coordinates in emissions from geo_em file
    geo_em_ds = xr.open_dataset(geo_em_fp)
    geo_em_ds = geo_em_ds.isel(Time=0)

    emissions_ds = xr.open_dataset(emissions_fp)#, chunks={'Time': 24})#, 'south_north': 10, 'west_east': 10})
    emissions_ds['XLONG'] = geo_em_ds.XLONG_M
    emissions_ds['XLAT'] = geo_em_ds.XLAT_M
    emissions_ds = emissions_ds.set_coords(['XLAT', 'XLONG'])
    emissions_ds.xoak.set_index(['XLAT', 'XLONG'], 'sklearn_geo_balltree')

    return emissions_ds
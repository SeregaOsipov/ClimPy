import xarray as xr
import pandas as pd
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
    emissions_ds = emissions_ds.rename({'south_north':'latitude', 'west_east':'longitude'})

    return emissions_ds

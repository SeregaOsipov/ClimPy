import xarray as xr
import pandas as pd
import xoak
from climpy.utils.wrf_utils import generate_xarray_uniform_time_data
import functools


def aggregate_variables_into_dim(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        aggregate_variables_into_dim = False
        if 'aggregate_variables_into_dim' in kwargs:
            aggregate_variables_into_dim = kwargs.pop('aggregate_variables_into_dim')

        ds = func(*args, **kwargs)

        if aggregate_variables_into_dim:
            das = [ds[key] for key in ds.data_vars]
            da = xr.concat(das, dim='species')
            da['species'] = list(ds.data_vars)
            da = da.rename('emissions')
            ds = da

        return ds
    return wrapper_decorator


@aggregate_variables_into_dim
def prep_wrf_emissions(fp):
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
    return emissions_ds



def prep_wrf_emissions_injecting_geo_em_coordinates(emissions_fp, geo_em_fp):
    # Supplements the missing the coordinates in emissions from geo_em file
    geo_em_ds = xr.open_dataset(geo_em_fp)
    geo_em_ds = geo_em_ds.isel(Time=0)

    emissions_ds = xr.open_dataset(emissions_fp)#, chunks={'Time': 24})#, 'south_north': 10, 'west_east': 10})
    emissions_ds['XLONG'] = geo_em_ds.XLONG_M
    emissions_ds['XLAT'] = geo_em_ds.XLAT_M
    emissions_ds = emissions_ds.set_coords(['XLAT', 'XLONG'])
    emissions_ds.xoak.set_index(['XLAT', 'XLONG'], 'sklearn_geo_balltree')

    return emissions_ds
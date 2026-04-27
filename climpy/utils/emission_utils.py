import functools
import xarray as xr
from climpy.utils.wrf_utils import fix_time_variable_in_wrf_output


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
def prep_wrf_emissions(fp, unify_time_variable=True, sel_and_squeeze_surface_layer=True):
    emissions_ds = xr.open_dataset(fp)
    if sel_and_squeeze_surface_layer:
        emissions_ds = emissions_ds.isel(emissions_zdim=0)  # Dropping the singleton dimension causes issues with WRF-Chem emission files. Drop it later in the scripts
    if unify_time_variable: # Don't rename time variable if the file will be used to run WRF-Chem
        emissions_ds = fix_time_variable_in_wrf_output(emissions_ds)
    return emissions_ds
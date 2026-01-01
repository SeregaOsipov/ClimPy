from datetime import datetime
import numpy as np
import climpy.utils.netcdf_utils as nc_utils
# from libs.readers import AbstractNetCdfReader as ancr
import matplotlib.pyplot as plt
import xarray as xr
import wrf as wrf
from wrf import Constants
import pandas as pd
from scipy import interpolate

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'


def column_averaging_1d_impl(wrf_da_1d, wrf_pressure_1d, tropomi_pressure_bins_1d):
    '''
    This function averages wrf profile between tropomi pressure levels
    :param wrf_da_1d:
    :param wrf_pressure_1d:
    :param tropomi_pressure_bins_1d:
    :return:
    '''
    if tropomi_pressure_bins_1d[0] == 0 or np.isnan(tropomi_pressure_bins_1d[0]):
        return np.full_like(tropomi_pressure_bins_1d, np.nan, dtype=float)

    # print(wrf_da_1d.shape)
    # print('wrf_da_1d')
    # print(wrf_da_1d)
    # print('wrf_pressure_1d')
    # print(wrf_pressure_1d)
    # print('tropomi_pressure_bins_1d')
    # print(tropomi_pressure_bins_1d)

    s = pd.Series(wrf_da_1d, index=wrf_pressure_1d)
    bins = pd.cut(s.index, bins=tropomi_pressure_bins_1d, include_lowest=True)
    binned_mean = s.groupby(bins).mean().values
    # return wrf_da_1d.groupby_bins('pressure', tropomi_da_1d)

    # return binned_mean
    return np.append(binned_mean, np.nan)  # have to keep a consistent size for xarray apply_ufunc


def average_wrf_diag_between_tropomi_staggered_pressure_grid(wrf_ds, wrf_key, tropomi_ds):
    # get wrf ch4 mixing ratio averaged between tropomi levels
    da = xr.apply_ufunc(column_averaging_1d_impl, wrf_ds[wrf_key], wrf_ds.pressure, tropomi_ds.p_stag.sel(level=slice(None, None, -1)),  # pressure must increase monotonically for group_by
                        input_core_dims=[['bottom_top'], ['bottom_top'], ['level']],
                        output_core_dims=[['layer']],
                        vectorize=True,
                        dask='parallelized',
                        # output_sizes={'new_p_levels': 12},  # default size of angles are not provided
                        dask_gufunc_kwargs={'output_sizes': {'layer': 12}},  # default size of angles are not provided
                        exclude_dims={'level'},
                        output_dtypes=[wrf_ds[wrf_key].dtype],
                        keep_attrs=True,
                        )
    da = da.isel(layer=slice(0, -1)) # drop last fake layer
    return da


def interpolation_in_pressure_1d_impl(wrf_da_1d, wrf_pressure_1d, tropomi_pressure_1d, tropomi_qa_value_1d):
    '''
    This function averages wrf profile to tropomi pressure levels
    :param wrf_da_1d:
    :param wrf_pressure_1d:
    :param tropomi_pressure_1d:
    :return:
    '''
    # if tropomi_pressure_1d[0] == 0 or np.isnan(tropomi_pressure_1d[0]):
    if tropomi_qa_value_1d == 0:
        return np.full_like(tropomi_pressure_1d, np.nan, dtype=float)

    # print('wrf_da_1d')
    # print(wrf_da_1d)
    # print('wrf_pressure_1d')
    # print(wrf_pressure_1d)
    # print('tropomi_pressure_1d')
    # print(tropomi_pressure_1d)
    # print('tropomi_qa_value_1d {}'.format(tropomi_qa_value_1d))

    interp_func = interpolate.interp1d(wrf_pressure_1d, wrf_da_1d, kind='linear', bounds_error=False, fill_value=np.nan)  # check for nans after interpolation
    return interp_func(tropomi_pressure_1d)


def interpolate_wrf_diag_to_tropomi_rho_pressure_grid(wrf_ds, wrf_key, tropomi_ds):
    print('Remember that interpolated DIAG profile will contain NaNs if TROPOMI top is above WRF top')
    da = xr.apply_ufunc(interpolation_in_pressure_1d_impl, wrf_ds[wrf_key], wrf_ds.pressure, tropomi_ds.p_rho, tropomi_ds.qa_value,
                        input_core_dims=[['bottom_top'], ['bottom_top'], ['layer'],[]],
                        output_core_dims=[['layer']],
                        vectorize=True,
                        dask='parallelized',
                        output_dtypes=[wrf_ds[wrf_key].dtype],
                        keep_attrs=True,
                        )
    return da


## PP section


def calculate_air_mass_dry(wrf_ds):
    '''
    Calculate dry air mass per unit area in kg / m^2
    Divide by dz to get air density


    FYI https://forum.mmm.ucar.edu/threads/how-to-calculate-grid-mu-with-a-corrected-surface-pressure.10927/

    The 3d dry pressure is defined as: p_dry(i,k,j) = eta(k) * mut(i,j) + ptop,
    where eta(k) is the WRF vertical coordinate,
    mut(i,j) is the dry column pressure,
    ptop is the model lid.

    If we are interested in the surface pressure, then the full level eta(k=1) is identically defined as 1.0.
    We know that mut(i,j) = mub(i,j) + mu(i,j),

    :param wrf_ds:
    :return:
    '''
    d_eta = -1 * wrf_ds.ZNW.diff(dim='bottom_top_stag').rename({'bottom_top_stag': 'bottom_top'})  # m
    d_eta.name = 'd_eta'
    d_eta.attrs['description'] = 'layers thickness in eta coordinates'
    d_eta.attrs['units'] = 'm'

    wrf_ds['air_mass_dry'] = (wrf_ds.MUB + wrf_ds.MU) / Constants.G * d_eta  # kg / m^2


def compute_dz(wrf_ds):
    wrf_ds['z_stag'] = (wrf_ds.PH+wrf_ds.PHB) / Constants.G

    wrf_ds['dz'] = wrf_ds.z_stag.diff(dim='bottom_top_stag').rename({'bottom_top_stag': 'bottom_top'})  # m
    wrf_ds.dz.name = 'dZ'
    wrf_ds.dz.attrs['description'] = 'layers thickness dZ'
    wrf_ds.dz.attrs['units'] = 'm'


def compute_stag_z(nc_in, time_index=None, squeeze=False):
    z_stag = wrf.getvar(nc_in, "zstag", timeidx=time_index, squeeze=squeeze)

    dz = z_stag.diff(dim='bottom_top_stag').rename({'bottom_top_stag': 'bottom_top'})  # m
    dz.name = 'dZ'
    dz.attrs['description'] = 'layers thickness dZ'
    dz.attrs['units'] = 'm'

    return z_stag, dz


def compute_p(wrf_ds):
    wrf_ds['pressure'] = wrf_ds.P + wrf_ds.PB  # Pa


def compute_stag_p(wrf_ds):
    '''
    wrf_ds should hold additional variables: z_stag, dz
    call first:
    compute_dz(wrf_ds)
    compute_p(wrf_ds)

    :param wrf_ds:
    :return:
    '''

    z_stag = wrf_ds.z_stag
    p_sfc = wrf_ds.PSFC
    pressure = wrf_ds.pressure

    pressure_stag = z_stag.copy(deep=True)
    pressure_stag.data = np.full_like(z_stag, np.nan)
    pressure_stag.loc[dict(bottom_top_stag=0)] = p_sfc
    for layer_index in pressure_stag.bottom_top_stag.values[:-1]:
        half_dp = pressure_stag.isel(bottom_top_stag=layer_index) - pressure.isel(bottom_top=layer_index)  # pressure_stag[:, layer_index] - pressure[:, layer_index]
        pressure_stag.loc[dict(bottom_top_stag=layer_index + 1)] = pressure_stag.isel(bottom_top_stag=layer_index) - half_dp * 2

    dp = -1 * pressure_stag.diff(dim='bottom_top_stag').rename({'bottom_top_stag': 'bottom_top'})
    dp.name = 'dP'
    dp.attrs['description'] = 'layers thickness dP'

    wrf_ds['pressure_stag'] = pressure_stag
    wrf_ds['dP'] = dp


def compute_stag_pressure(nc_in, time_index=None, squeeze=False):
    pressure = wrf.getvar(nc_in, "pressure", timeidx=time_index, squeeze=False)
    p_sfc = wrf.getvar(nc_in, 'PSFC', timeidx=time_index, squeeze=False) / 10 ** 2
    z_stag, dz = compute_stag_z(nc_in, time_index, squeeze)
    pressure_stag, dp = compute_stag_pressure_impl(pressure, p_sfc, z_stag)

    return pressure, pressure_stag, p_sfc, dp, z_stag, dz


def compute_stag_pressure_impl(pressure, p_sfc, z_stag):
    pressure_stag = z_stag.copy(deep=True)
    pressure_stag.data = np.full_like(z_stag, np.nan)
    pressure_stag.loc[dict(bottom_top_stag=0)] = p_sfc
    for layer_index in pressure_stag.bottom_top_stag.values[:-1]:
        half_dp = pressure_stag.isel(bottom_top_stag=layer_index) - pressure.isel(bottom_top=layer_index)  # pressure_stag[:, layer_index] - pressure[:, layer_index]
        pressure_stag.loc[dict(bottom_top_stag=layer_index + 1)] = pressure_stag.isel(bottom_top_stag=layer_index) - half_dp * 2

    dp = -1 * pressure_stag.diff(dim='bottom_top_stag').rename({'bottom_top_stag': 'bottom_top'})
    dp.name = 'dP'
    dp.attrs['description'] = 'layers thickness dP'

    return pressure_stag, dp


# Utils Section


def inject_geo_em_coordinates(ds, geo_em_fp):
    # Supplements the missing the coordinates in emissions from geo_em file
    geo_em_ds = xr.open_dataset(geo_em_fp)
    geo_em_ds = geo_em_ds.isel(Time=0)

    ds['XLONG'] = geo_em_ds.XLONG_M
    ds['XLAT'] = geo_em_ds.XLAT_M
    ds = ds.set_coords(['XLAT', 'XLONG'])  # TODO: make use of it everywhere
    # ds.xoak.set_index(['XLAT', 'XLONG'], 'sklearn_geo_balltree')

    return ds


def plot_domain(nc, subplot_config=111):
    """
    quick way to plot the WRF simulation domain
    :param nc:
    :return:
    """
    # import to work with WRF output
    import wrf as wrf  # wrf-python library https://wrf-python.readthedocs.io/en/latest/

    projection = wrf.get_cartopy(wrfin=nc)  # Set the GeoAxes to the projection used by WRF
    ax = plt.subplot(subplot_config, projection=projection)  # axes
    ax.coastlines('50m', linewidth=0.8)

    # Set the map bounds
    ax.set_xlim(wrf.cartopy_xlim(wrfin=nc))
    ax.set_ylim(wrf.cartopy_ylim(wrfin=nc))

    # Add the gridlines
    # ax.gridlines(color="black", linestyle="dotted")
    # gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False)

    return ax


def derive_wrf_net_flux_from_accumulated_value(nc, acc_down_name, acc_up_name, tyxSlicesArray, dt):
    down_accumulated = nc.variables[acc_down_name][tyxSlicesArray]
    up_accumulated = nc.variables[acc_up_name][tyxSlicesArray]

    down = down_accumulated[1:] - down_accumulated[:-1]
    up = up_accumulated[1:] - up_accumulated[:-1]

    # convert J to W
    dt_second = dt.days*24*60*60+dt.seconds
    down /= dt_second
    up /= dt_second
    net = down - up

    # net_masked = np.ma.array(np.empty(down_accumulated.shape))
    # net_masked[0] = np.NaN
    # net_masked.mask = land_mask[np.newaxis, :, :]
    # net_masked.data[1:] = net

    return net, down, up


def derive_wrf_net_flux_from_instanteneous_value(nc, down_name, up_name, tyxSlicesArray, dt): #, land_mask, daily_mean_number_of_steps):
    down_flux = nc.variables[down_name][tyxSlicesArray]
    up_flux = nc.variables[up_name][tyxSlicesArray]
    net_flux = down_flux - up_flux

    # net_masked = np.ma.array(net_flux)
    # net_masked.mask = land_mask[np.newaxis, :, :]

    # down_masked = np.ma.array(down_flux)
    # down_masked.mask = land_mask[np.newaxis, :, :]

    # up_masked = np.ma.array(up_flux)
    # up_masked.mask = land_mask[np.newaxis, :, :]

    # return compute_wrf_daily_data(net_masked, daily_mean_number_of_steps), compute_wrf_daily_data(down_masked, daily_mean_number_of_steps), compute_wrf_daily_data(up_masked, daily_mean_number_of_steps)

    return net_flux, down_flux, up_flux


def compute_wrf_daily_data(data, daily_mean_number_of_steps):
    data = data.reshape((-1, daily_mean_number_of_steps) + data.shape[1:])
    data = np.nanmean(data, axis=1)
    return data


def prepare_wrf_net_flux(nc, acc_down_name, acc_up_name, tyxSlicesArray, land_mask, daily_mean_number_of_steps):
    down = nc.variables[acc_down_name][tyxSlicesArray]
    up = nc.variables[acc_up_name][tyxSlicesArray]
    net = down - up
    # convert J to W
    net /= 3 * 60 * 60

    net = compute_wrf_daily_data(net, daily_mean_number_of_steps)
    net = np.ma.array(net)
    net.mask = land_mask[np.newaxis, :, :]

    return net


def generate_netcdf_uniform_time_data(time_variable, td1=None, td2=None):
    if td1 is None:
        # td1 = dt.datetime.strptime(str(netCDF4.chartostring(time_variable[0])), '%Y-%m-%d_%H:%M:%S')
        td1 = datetime.strptime(''.join([char.decode("utf-8") for char in time_variable[0]]), '%Y-%m-%d_%H:%M:%S')
    if td2 is None:
        td2 = datetime.strptime(''.join([char.decode("utf-8") for char in time_variable[1]]), '%Y-%m-%d_%H:%M:%S')
    rawTimeData = nc_utils.generate_netcdf_uniform_time_data(time_variable, td1, td2)

    return rawTimeData


def generate_xarray_uniform_time_data(time_variable, td1=None, td2=None):
    if td1 is None:
        td1 = datetime.strptime(time_variable[0].values.tostring().decode("utf-8"), '%Y-%m-%d_%H:%M:%S')
    if td2 is None:
        td2 = datetime.strptime(time_variable[1].values.tostring().decode("utf-8"), '%Y-%m-%d_%H:%M:%S')
    rawTimeData = nc_utils.generate_netcdf_uniform_time_data(time_variable, td1, td2)

    return rawTimeData


def create_times_var(dates):
    """
    Convert dates to WRF Times string

    Write like this:
    dates = pd.date_range(dt.datetime(2050, 1, 1), dt.datetime(2050, 1, 1) + dt.timedelta(hours=1*nc['Times'].shape[0]-1),freq='h')
    aux_times = create_times_var(dates)
    nc.variables['Times'][:] = aux_times
    nc.close()

    :return:
    """

    aux_times = np.chararray((len(dates), 19), itemsize=1)
    for i, date in enumerate(dates):
        aux_times[i] = list(date.strftime("%Y-%m-%d_%H:%M:%S"))
    return aux_times


def derive_wrf_cells_area(ds):
    cell_area = ds.DX / ds.MAPFAC_MX * ds.DY /ds.MAPFAC_MY
    return cell_area


def derive_wrf_pressure_from_met_and_input_files(wrf_met_ds, wrf_ic_ds):
    with xr.set_options(keep_attrs=True):
        wrf_met_ds['p_rho'] = wrf_ic_ds.P_TOP + (wrf_met_ds.PSFC - wrf_ic_ds.P_TOP) * wrf_ic_ds.ZNU
    wrf_met_ds.p_rho.attrs['description'] = 'Pressure at eta values on half (mass) levels'

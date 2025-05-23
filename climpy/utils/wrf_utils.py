from datetime import datetime
import numpy as np
import climpy.utils.netcdf_utils as nc_utils
# from libs.readers import AbstractNetCdfReader as ancr
import matplotlib.pyplot as plt
import xarray as xr

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

Z_DIM_NC_KEY = 'bottom_top'


def inject_geo_em_coordinates(ds, geo_em_fp):
    # Supplements the missing the coordinates in emissions from geo_em file
    geo_em_ds = xr.open_dataset(geo_em_fp)
    geo_em_ds = geo_em_ds.isel(Time=0)

    ds['XLONG'] = geo_em_ds.XLONG_M
    ds['XLAT'] = geo_em_ds.XLAT_M
    # ds = ds.set_coords(['XLAT', 'XLONG'])  # TODO: make use of it everywhere
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


def get_cell_area(ds):
    # defmapfactor_mx = ds.variables['MAPFAC_MX'][0]
    # mapfactor_my = ds.variables['MAPFAC_MY'][0]
    # dx = getattr(ds, 'DX')  # m
    # dy = getattr(ds, 'DY')
    # cell_area = np.ones(mapfactor_mx.shape) * dx / mapfactor_mx * dy / mapfactor_my

    cell_area = ds.DX / ds.MAPFAC_MX * ds.DY /ds.MAPFAC_MY
    return cell_area
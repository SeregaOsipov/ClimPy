import functools
import numpy as np
import matplotlib

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'


def time_averaging(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        vo = func(*args, **kwargs)
        vo['data'] = np.nanmean(vo['data'], axis=(0,))
        # vo['time'] # shall it be averaged?
        return vo
    return wrapper_decorator


def time_interval_selection(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        time_range = None
        if 'time_range' in kwargs:
            time_range = kwargs.pop('time_range')
        time_dim = 0
        if 'time_dim' in kwargs:
            time_dim = kwargs.pop('time_dim')

        vo = func(*args, **kwargs)

        if time_range is not None:
            ind = np.logical_and(vo['time'] >= time_range[0], vo['time'] <= time_range[1])
            # if coordinates are time dependent, apply index too
            # make sure to check the size first, and only then adjust it
            if 'lat' in vo.keys() and vo['data'].shape[0] == vo['lat'].shape[0]:
                # TODO: check that lat and lon are not time dependant
                vo['lat'] = vo['lat'][ind]
                vo['lon'] = vo['lon'][ind]
            if 'level' in vo.keys() and vo['level'].shape[0] == vo['data'].shape[0]:
                vo['level'] = vo['level'][ind]
            # vo['data'] = vo['data'][ind]
            vo['data'] = np.compress(ind, vo['data'], axis=time_dim)
            vo['time'] = vo['time'][ind]
        return vo
    return wrapper_decorator


def pandas_time_interval_selection(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        time_range = None
        if 'time_range' in kwargs:
            time_range = kwargs.pop('time_range')

        df = func(*args, **kwargs)

        if time_range is not None:
            ind = np.logical_and(df.index >= time_range[0], df.index <= time_range[1])
            df = df.loc[ind]
        return df
    return wrapper_decorator


def temporal_sampling(func):
    '''
    This decorator reduces the temporal sampling of the data
    :param func:
    :return:
    '''
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        sampling_step = None
        if 'sampling_step' in kwargs:
            sampling_step = kwargs.pop('sampling_step')

        vo = func(*args, **kwargs)

        if sampling_step is not None:
            vo['time'] = vo['time'][::sampling_step]
            vo['lat'] = vo['lat'][::sampling_step]
            vo['lon'] = vo['lon'][::sampling_step]
            vo['data'] = vo['data'][::sampling_step]
        return vo
    return wrapper_decorator


def convert_dates_to_years_since_first_date(func):
    '''
    # this will convert dates to years since the first date
    :param func:
    :return:
    '''
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        vo = func(*args, **kwargs)
        time_data = matplotlib.dates.date2num(np.copy(vo['time']))  # convert to days since 1-1-1
        time_data -= time_data[0]
        time_data /= 365
        vo['time'] = time_data
        return vo
    return wrapper_decorator


def lonlat_weight_averaging(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        weights = None
        if 'surface_area' in kwargs:
            weights = kwargs.pop('surface_area')
        else:
            raise Exception('lonlat_weight_averaging', 'weight for spatial averaging is missing')

        vo = func(*args, **kwargs)

        # if the pressure is 4d, weight it too
        if vo['level'].shape[-2:] == vo['data'].shape[-2:]:
            vo['level'] = np.sum(vo['level'] * weights, axis=(-2, -1)) / np.sum(weights)

        vo['data'] = np.sum(vo['data'] * weights, axis=(-2, -1)) / np.sum(weights)
        return vo

    return wrapper_decorator


def lon_averaging(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        vo = func(*args, **kwargs)
        vo['data'] = np.mean(vo['data'], axis=vo['lon_axis'])
        # vo['lon'] # shall it be averaged?
        return vo
    return wrapper_decorator


def compress_1d(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        compress_1d_ind = None
        compress_1d_axis = None
        if 'compress_1d_ind' in kwargs and 'compress_1d_axis' in kwargs:
            compress_1d_ind = kwargs.pop('compress_1d_ind')
            compress_1d_axis = kwargs.pop('compress_1d_axis')

        vo = func(*args, **kwargs)

        if compress_1d_axis is not None and compress_1d_ind is not None:
            # if the pressure is 4d, compress it too
            if vo['level'].shape[compress_1d_axis] == vo['data'].shape[compress_1d_axis]:
                vo['level'] = np.compress(compress_1d_ind, vo['level'], compress_1d_axis)
            vo['data'] = np.compress(compress_1d_ind, vo['data'], compress_1d_axis)
        return vo

    return wrapper_decorator


def data_statistics(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        vo = func(*args, **kwargs)
        print('min is {:.2e}, max is {:.2e}'.format(np.min(vo['data']), np.max(vo['data'])))
        return vo
    return wrapper_decorator


def lexical_preprocessor(func):
    """'
    This decorator helps to deal with cases, when you just want to get diag1+diag2 in one line
    For example, Ox production + Ox destruction.
    The decorator splits by + and loops over the keys.
    """
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        var_key = args[1]
        if '+' in var_key:
            print("Lexical preprocessor split {} into + keys {}".format(var_key, var_key.split('+')))
            vos = []
            total = 0
            for key in var_key.split('+'):
                # args[1] = key
                # vo = func(*args, **kwargs)
                vo = func(args[0], key, **kwargs)
                vos.append(vo)
                total += vo['data']
            vo['data'] = total
        else:
            vo = func(*args, **kwargs)
        return vo
    return wrapper_decorator


@lexical_preprocessor
def get_diag_template(model_vo, var_key, anomaly_wrt_vo=None, is_anomaly_relative=False):
    vo = {}
    vo['data'] = model_vo[var_key]
    vo['lon'], vo['lat'] = model_vo['lon'], model_vo['lat']
    vo['time'] = model_vo['time']
    vo['level'] = model_vo['p_rho']

    if var_key+'_units' in model_vo.keys():
        vo['units'] = model_vo[var_key+'_units']

    if anomaly_wrt_vo is not None:
        # data can have different temporal extent
        t_min = np.min((model_vo[var_key].shape[0], anomaly_wrt_vo[var_key].shape[0]))
        t_slice = slice(0, t_min)

        if vo['level'].shape[0] == vo['data'].shape[0]:
            vo['level'] = vo['level'][t_slice]
        vo['data'] = model_vo[var_key][t_slice] - anomaly_wrt_vo[var_key][t_slice]
        vo['time'] = vo['time'][t_slice]

        if is_anomaly_relative:
            vo['data'] /= anomaly_wrt_vo[var_key][t_slice]

    return vo


def normalize_size_distribution_by_point(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        sd_normalization_raduis = None  # wavelength for example 1 um
        if 'sd_normalization_raduis' in kwargs:
            sd_normalization_raduis = kwargs.pop('sd_normalization_raduis')

        sd_ds = func(*args, **kwargs)

        if sd_normalization_raduis is not None:
            # ind = sd_ds['radii'] == sd_normalization_raduis
            # if not np.any(ind):
            # print('Cant find exact radii for normalization')
            distance = np.abs((sd_ds['radii'] - sd_normalization_raduis))
            ind = distance.argmin()
            # print('Instead searched for the closest {} given {}'.format(sd_ds['radii'][ind], sd_normalization_raduis))

            scale = 1 / sd_ds['data'][:, ind]  # first dimension is time, second radius
            sd_ds['data'] *= scale[:, np.newaxis]
        return sd_ds
    return wrapper_decorator


def normalize_size_distribution_by_area(func):
    '''
    This decorator will normalize the area under the curve in the given range to 1

    :param func:
    :param normalization_diameters: [m]
    :return:
    '''

    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        normalization_diameters = None  # range of diameters for normalization, meters
        diameter_dim = None
        size_modes_dim = None
        if 'normalization_diameters' in kwargs:
            normalization_diameters = kwargs.pop('normalization_diameters')
            diameter_dim = kwargs.pop('diameter_dim')
        if 'size_modes_dim' in kwargs:
            size_modes_dim = kwargs.pop('size_modes_dim')

        vo = func(*args, **kwargs)

        if normalization_diameters is not None:
            dp = vo['radii'] * 2 * 10**-6  # m
            dp = np.squeeze(dp)
            ind = np.logical_and(dp >= normalization_diameters[0], dp <= normalization_diameters[1])
            logdp = np.log(dp)

            if size_modes_dim is not None:  # if modes (nu, acc, coarse) are stored separately
                sd = np.sum(vo['data'], axis=size_modes_dim)
            else:
                sd = vo['data']

            area = np.trapz(np.compress(ind, sd, axis=diameter_dim), logdp[ind], axis=diameter_dim)
            vo['data'] /= np.expand_dims(area, axis=diameter_dim)

            # check normalization
            # np.trapz(vo['data'], logd, axis=1)
            # np.trapz(np.mean(vo['data'], axis=0), logd)
        return vo
    return wrapper_decorator


def derive_size_distribution_moment(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        moment = None  # 'dN' by default
        if 'moment' in kwargs:
            moment = kwargs.pop('moment')

        sd_ds = func(*args, **kwargs)

        if moment is not None:
            if moment == 'dV':  # compute dV from dN
                sd_ds['dNdlogd'] *= 4 / 3 * np.pi * sd_ds['radius'] ** 3  # *= um**3
                sd_ds = sd_ds.rename({'dNdlogd': 'dVdlogd'})
            if moment == 'dA':  # compute dA from dN, geometric cross-section (not surface area)
                sd_ds['dNdlogd'] *= np.pi * sd_ds['radius'] ** 2  # *= um**2
                sd_ds = sd_ds.rename({'dNdlogd': 'dAdlogd'})
            if moment == 'dS':  # compute dS from dN, surface area (not geometric cross-section)
                sd_ds['dNdlogd'] *= 4 * np.pi * sd_ds['radius'] ** 2  # *= um**2
                sd_ds = sd_ds.rename({'dNdlogd': 'dSdlogd'})
        return sd_ds
    return wrapper_decorator
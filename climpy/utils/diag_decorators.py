__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

import functools
import numpy as np
import matplotlib


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

        vo = func(*args, **kwargs)

        if time_range is not None:
            ind = np.logical_and(vo['time'] >= time_range[0], vo['time'] <= time_range[1])
            vo['data'] = vo['data'][ind]
            vo['time'] = vo['time'][ind]
            # if coordinates are time dependent, apply index too
            if 'lat' in vo.keys() and vo['data'].shape[0] == vo['lat'].shape[0]:
                # TODO: check that lat and lon are not time dependant
                vo['lat'] = vo['lat'][ind]
                vo['lon'] = vo['lon'][ind]
        return vo
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


def lon_averaging(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        vo = func(*args, **kwargs)
        vo['data'] = np.nanmean(vo['data'], axis=vo['lon_axis'])
        # vo['lon'] # shall it be averaged?
        return vo
    return wrapper_decorator


def data_statistics(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        vo = func(*args, **kwargs)
        print('min is {:.2e}, max is {:.2e}'.format(np.min(vo['data']), np.max(vo['data'])))
        return vo
    return wrapper_decorator


def get_diag_template(model_vo, var_key, anomaly_wrt_vo=None, is_anomaly_relative=False):
    vo = {}
    vo['data'] = model_vo[var_key]
    vo['lon'], vo['lat'] = model_vo['lon'], model_vo['lat']
    vo['time'] = model_vo['time']

    if anomaly_wrt_vo is not None:
        # data can have different temporal extent
        t_min = np.min((model_vo[var_key].shape[0], anomaly_wrt_vo[var_key].shape[0]))
        t_slice = slice(0, t_min)

        vo['data'] = model_vo[var_key][t_slice] - anomaly_wrt_vo[var_key][t_slice]
        vo['time'] = vo['time'][t_slice]
        if is_anomaly_relative:
            vo['data'] /= anomaly_wrt_vo[var_key][t_slice]

    return vo


def normalize_size_distribution(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        sd_normalization_raduis = None  # wavelength for example 1 um
        if 'sd_normalization_raduis' in kwargs:
            sd_normalization_raduis = kwargs.pop('sd_normalization_raduis')

        vo = func(*args, **kwargs)

        if sd_normalization_raduis is not None:
            # ind = vo['radii'] == sd_normalization_raduis
            # if not np.any(ind):
            # print('Cant find exact radii for normalization')
            distance = np.abs((vo['radii'] - sd_normalization_raduis))
            ind = distance.argmin()
            # print('Instead searched for the closest {} given {}'.format(vo['radii'][ind], sd_normalization_raduis))

            scale = 1 / vo['data'][:, ind]  # first dimension is time, second radius
            vo['data'] *= scale[:, np.newaxis]
        return vo
    return wrapper_decorator
import os as os
from matplotlib import pyplot as plt
import netCDF4
import numpy as np
from climpy.utils import wrf_utils as wrf_utils
import pandas as pd

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'


MY_DPI = 300


def JGR_page_width_inches():
    return 190 / 25.4


def screen_width_inches():
    return 1920 / MY_DPI


def save_figure_bundle(root_folder, file_name, vector_graphics=False):
    """
    Saves figure in 3 formats, png dpi 600, svg and pdf
    :return:
    """
    save_figure(root_folder, file_name, file_ext='png', dpi=300)
    save_figure(root_folder, file_name + ' dpi 600', file_ext='png', dpi=600)  # I need it for faster latex compilation
    if vector_graphics:
        save_figure(root_folder, file_name, file_ext='svg')
        save_figure(root_folder, file_name, file_ext='pdf')


def save_figure(root_folder, file_name, file_ext='png', dpi=MY_DPI):
    full_file_path = root_folder + file_name + '.' + file_ext
    dir_path = os.path.dirname(full_file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.savefig(full_file_path, dpi=dpi, facecolor='w', transparent=False)  # last 2 fix the transparent backgroud issue in jupyter


def plot_y_x_on_scatter_plot(ax, plot_2x=False):
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, 'k-', zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    if plot_2x:  # log plots some time need y=2x & y=0.5x
        ax.plot(lims, 2 * np.array(lims), 'k--', linewidth=0.5, zorder=0)
        ax.plot(lims, 1/2 * np.array(lims), 'k--', linewidth=0.5, zorder=0)


def plot_aeronet_stations_over_wrf_domain(wrf_nc_file_path, stations):
    '''
    plot the map with Aeronet stations
    :param wrf_nc_file_path:
    :param stations:
    :return:
    '''

    nc = netCDF4.Dataset(wrf_nc_file_path)
    fig = plt.figure(figsize=(12, 6))
    ax = wrf_utils.plot_domain(nc)
    import cartopy.crs as crs

    # marker
    ax.plot(stations['Longitude(decimal_degrees)'], stations['Latitude(decimal_degrees)'],
            color='orange', marker='o', linewidth=0,
            transform=crs.PlateCarree(),)

    # labels
    text_color = 'tab:orange'
    xo = 0.75
    yo = -0.35
    for index, station in stations.iterrows():
        ax.text(station['Longitude(decimal_degrees)'] + xo, station['Latitude(decimal_degrees)'] + yo, station['Site_Name'],
                color=text_color, transform=crs.PlateCarree(),)


def collocate_time_series(obs_var, obs_time, model_var, model_time, freq='10min', drop_nans=True, drop_zeros=False):
    '''
    Bin two datasets into time intervals
    Usefull for scatter plots
    '''
    obs_rounded_time = pd.Series(obs_time).dt.round(freq=freq)
    model_rounded_time = pd.Series(model_time).dt.round(freq=freq)
    # sample both data sets at the same time
    d1, obs_ind, model_ind = np.intersect1d(obs_rounded_time, model_rounded_time, return_indices=True)

    collocated_obs_var = obs_var[obs_ind]
    collocated_model_var = model_var[model_ind]

    # also drop nans
    if drop_nans:
        ind = np.logical_or(np.isnan(collocated_obs_var), np.isnan(collocated_model_var))
        ind = np.logical_not(ind)
        collocated_obs_var = collocated_obs_var[ind]
        collocated_model_var = collocated_model_var[ind]

    # also drop zeros
    if drop_zeros:
        ind = np.logical_or(collocated_obs_var == 0, collocated_model_var == 0)
        ind = np.logical_not(ind)
        collocated_obs_var = collocated_obs_var[ind]
        collocated_model_var = collocated_model_var[ind]

    return collocated_obs_var, obs_ind, collocated_model_var, model_ind


def get_default_colors_list():
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    return colors



def new_figure_1_3_impl():
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(1.5 * JGR_page_width_inches(), JGR_page_width_inches()), dpi=MY_DPI, constrained_layout=True)
    return fig, axes


def new_figure_3_3_impl(figsize=(1.5 * JGR_page_width_inches(), JGR_page_width_inches())):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=figsize, constrained_layout=True)  # , dpi=MY_DPI
    return fig, axes
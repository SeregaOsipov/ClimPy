import os as os
from matplotlib import pyplot as plt
import netCDF4
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import cartopy
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


def collocate_time_series(obs_ps, model_ps, freq='10min', drop_zeros=False):
    '''
    Bin two datasets into time intervals
    Usefull for scatter plots
    '''

    # obs_rounded_time = obs_ps.index.round(freq=freq)
    # model_rounded_time = model_ps.index.round(freq=freq)
    print('Skipping Collocation')

    # also drop zeros
    if drop_zeros:
        obs_ps = obs_ps.loc[(obs_ps!=0)]
        model_ps = model_ps.loc[(model_ps != 0)]

    common_index = obs_ps.index.intersection(model_ps.index)
    collocated_obs_ps = obs_ps[common_index]
    collocated_model_ps = model_ps[common_index]

    return collocated_obs_ps, collocated_model_ps


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


def get_manual_colormap(name):
    palette = None

    match name:
        case 'GEE_bbpcgyr':
            palette = ['black', 'blue', 'purple', 'cyan', 'green', 'yellow', 'red']
        case 'GEE_wbpcgyr':
            palette = ['white', 'blue', 'purple', 'cyan', 'green', 'yellow', 'red']  # drop black as it makes difficult to see coastlines
        case 'emissions':
            palette = ['white', '#A7CAB7', 'cyan', '#2C5F2D', 'yellow', 'red', 'brown']
        case 'emissions_v2':
            palette = ['#EEEEEE', '#A7CAB7', 'cyan', '#2C5F2D', 'yellow', 'red', 'brown']

    cm = LinearSegmentedColormap.from_list('GEE', palette, N=256)

    return cm


# Section: Maps with projections


def prepare_map_axes(ax):
    ax.coastlines()
    ax.add_feature(cartopy.feature.BORDERS, linestyle=':', linewidth=0.5)
    
    
def generate_logspaced_bins(min_value, max_value, bins_per_decade=5):
    '''
    Generate logspaced bins, so that 1 order of magnitude has specified number of bins
    '''
    if min_value <= 0 or max_value <= min_value:
        raise ValueError("min_value must be positive and max_value must be greater than min_value")
    
    log_min = np.log10(min_value)
    log_max = np.log10(max_value)
    num_decades = log_max - log_min
    num_points = int(np.ceil(num_decades * bins_per_decade)) + 1
    bins = np.logspace(log_min, log_max, num_points)
    return bins
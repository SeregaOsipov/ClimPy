import os as os
from matplotlib import pyplot as plt
import netCDF4
__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

from climpy.utils import wrf_utils as wrf_utils

MY_DPI = 96.0


def JGR_page_width_inches():
    return 190 / 25.4


def screen_width_inches():
    return 1920 / MY_DPI


def save_figure_bundle(root_folder, file_name):
    """
    Saves figure in 3 formats, png dpi 600, svg and pdf
    :return:
    """
    save_figure(root_folder, file_name, file_ext='png', dpi=600)
    save_figure(root_folder, file_name, file_ext='svg')
    save_figure(root_folder, file_name, file_ext='pdf')


def save_figure(root_folder, file_name, file_ext='png', dpi=MY_DPI):
    full_file_path = root_folder + file_name + '.' + file_ext
    dir_path = os.path.dirname(full_file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.savefig(full_file_path, dpi=dpi)


def plot_aeronet_stations_over_wrf_domain(wrf_nc_file_path, stations):
    '''
    plot the map with Aeronet stations
    :param wrf_nc_file_path:
    :param stations:
    :return:
    '''

    nc = netCDF4.Dataset(wrf_nc_file_path)
    fig, ax = wrf_utils.plot_domain(nc)
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
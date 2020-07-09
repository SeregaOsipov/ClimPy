__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

import os as os
from matplotlib import pyplot as plt


MY_DPI = 96.0


def get_JGR_full_page_width_inches():
    return 190 / 25.4


def get_full_screen_page_width_inches():
    return 1920 / MY_DPI


def save_figure_bundle(root_folder, file_name):
    """
    Saves figure in 3 formats, png dpi 600, svg and pdf
    :return:
    """
    save_fig(root_folder, file_name + '.png', dpi=600)
    save_fig(root_folder, file_name + '.svg')
    save_fig(root_folder, file_name + '.pdf')


def save_fig(root_folder, file_name, dpi=MY_DPI):
    full_file_path = root_folder + file_name
    dir_path = os.path.dirname(full_file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.savefig(full_file_path, dpi=dpi)

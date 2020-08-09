import os as os
from matplotlib import pyplot as plt

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

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

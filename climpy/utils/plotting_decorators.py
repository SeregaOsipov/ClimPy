import functools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from climpy.utils.plotting_utils import screen_width_inches, MY_DPI
from scipy.constants import golden

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'


def create_new_figure(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        fig = plt.figure(constrained_layout=True, figsize=(screen_width_inches() / 2, screen_width_inches() / 2 /golden), dpi=MY_DPI)

        vo = func(*args, **kwargs)
        return vo
    return wrapper_decorator
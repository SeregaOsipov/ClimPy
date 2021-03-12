import numpy as np

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'


def get_mosaic_bins():
    dlo_um = 0.039063e-6  # m
    dhi_um = 10e-6  # m
    d_stag = np.logspace(np.log(dlo_um), np.log(dhi_um), 9, base=np.e)
    return d_stag


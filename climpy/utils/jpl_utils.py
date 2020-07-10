import os
import numpy as np
from climpy.utils.file_path_utils import get_root_storage_path_on_hpc

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'


"""
ported from my MATLAB scripts
"""


def parse_absorption_cross_section_file(species_file_name):
    """
    parses the abs cross section file obtained from the MPI atlas
    :param species_file_name:
    :return:
    """

    # mpi_atlas_root_dir = os.path.expanduser('~') + '/Data/MPI/atlas/'
    mpi_atlas_root_dir = get_root_storage_path_on_hpc() + '/Data/MPI/atlas/'

    raw_data = np.genfromtxt(mpi_atlas_root_dir + 'xSection/' + species_file_name)

    if raw_data.shape[1] > 2:
        # this is the O3 case, where they put the '-' symbol for the wavelengths range
        wl_left = raw_data[:, 0]
        wl_right = raw_data[:, 2]
        abs_xsection = raw_data[:, 3]

        wl_rho = (wl_left + wl_right)/2
    else:
        wl_rho = raw_data[:, 0]
        abs_xsection = raw_data[:, 1]

    xs_vo = {}
    xs_vo['wavelengths'] = wl_rho / 10**3  # nm -> um
    xs_vo['abs_xsection'] = abs_xsection  # cm^2 * molecule^-1

    return xs_vo
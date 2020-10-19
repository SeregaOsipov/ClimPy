import copy
from climpy.utils.file_path_utils import get_pictures_root_folder
from climpy.utils.sparc_asap_sato_cmip_utils import prepare_sparc_asap_profile_data, disassemble_sparc_into_diags, \
    derive_sparc_so4_wet_mass
from climpy.utils.plotting_utils import save_figure
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

pics_output_folder = get_pictures_root_folder() + '/Papers/PinatuboInitialStage/SparcAsap/'

sparc_profile_vo = prepare_sparc_asap_profile_data(True, filter_unphysical_data=True)
r_eff_vo, ext_1020_vo, aod_1020_vo = disassemble_sparc_into_diags(sparc_profile_vo)
so4_mass_vo = derive_sparc_so4_wet_mass(r_eff_vo, ext_1020_vo)


def get_diag(vo, extra_weight=None, lat_limits=[-90, 90]):
    '''
    var_index: 1 is 1020 nm ext, -7 is r_eff
    '''

    ind = np.isnan(vo['data'])

    weight = np.cos(np.deg2rad(vo['lat']))
    if extra_weight is not None:
        weight *= extra_weight  # include 1020 extinction as weight too
        ind = np.logical_or(ind, np.isnan(extra_weight))  # r_eff and ext_1020 masks do not coincide

    print('{} NaN values were ignored'.format(np.sum(ind)))

    if lat_limits is not None:
        lat_ind = np.logical_and(vo['lat'] >= lat_limits[0], vo['lat'] <= lat_limits[1])
        lat_ind = np.logical_not(lat_ind)  # True values will be masked
        ind = np.logical_or(ind, lat_ind)

    # use masked array to properly average the data
    data_mx = np.ma.masked_array(vo['data'], mask=ind)
    # use the same mask for weight
    weight_mx = np.ma.masked_array(weight, mask=ind)
    # average
    axes = tuple(range(data_mx.ndim))[1:]  # all, except time
    result = np.sum(data_mx * weight_mx, axis=axes) / np.sum(weight_mx, axis=axes)

    vo = copy.deepcopy(vo)
    vo['data'] = result

    return vo


# Global plots: r_eff, aod, mass

def plot_r_ext_mass(lat_limits):
    plt.figure(constrained_layout=True, figsize=(10, 5))
    plt.clf()

    ax = plt.gca()
    color = 'k'
    diag_vo = get_diag(r_eff_vo, extra_weight=ext_1020_vo['data'], lat_limits=lat_limits)
    plt.plot(diag_vo['time'], diag_vo['data'], '-o', ms=4, color=color, label='Effective radius')
    plt.ylabel('Radius, ($\mu m$)')
    plt.xlabel('Time, ()')

    color = 'tab:orange'
    ax2 = ax.twinx()
    ax2.tick_params(axis='y', labelcolor=color)
    diag_vo = get_diag(aod_1020_vo, lat_limits=lat_limits)
    plt.plot(diag_vo['time'], diag_vo['data'], '-*', ms=8, color=color, label='Optical depth')
    plt.ylabel('Optical depth, ()', color=color)

    color = 'tab:blue'
    ax3 = ax.twinx()
    ax3.spines["right"].set_position(("axes", 1.1))
    ax3.tick_params(axis='y', labelcolor=color)

    vo = so4_mass_vo
    lat_ind = np.logical_and(vo['lat'] >= lat_limits[0], vo['lat'] <= lat_limits[1])
    data_mx = np.ma.masked_array(vo['data'], mask=np.logical_not(lat_ind))
    data_to_plot = np.nansum(data_mx, axis=(1, 2))
    plt.plot(diag_vo['time'], data_to_plot, '-o', ms=4, color=color, label='Mass')
    plt.ylabel('Mass, (kg)', color=color)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    h3, l3 = ax3.get_legend_handles_labels()
    ax2.legend(h1+h2+h3, l1+l2+l3, loc='upper right')


plt.ion()

# global
lat_limits = [-90, 90]
for lat_limits in ([-90, 90], [-20, 20]):
    plot_r_ext_mass(lat_limits)
    plt.title('Latitude band [{}, {}]'.format(lat_limits[0], lat_limits[1]))
    plt.yscale('linear')
    plt.draw()

    sub_folder = '/lat [{}, {}]/'.format(lat_limits[0], lat_limits[1])
    save_figure(pics_output_folder + sub_folder, 'sparc asap r_eff ext_1020 mass')
    plt.yscale('log')
    save_figure(pics_output_folder + sub_folder, 'sparc asap r_eff ext_1020 log mass')

    plt.xlim(dt.datetime(1990,1,1), dt.datetime(1996,1,1))
    plt.yscale('linear')
    save_figure(pics_output_folder + sub_folder, 'sparc asap r_eff ext_1020 mass 1990-1996')
    plt.yscale('log')
    save_figure(pics_output_folder + sub_folder, 'sparc asap r_eff ext_1020 log mass 1990-1996')





plt.ion()
plt.figure(constrained_layout=True, figsize=(10, 5))
plt.clf()
r_eff_vo = get_diag(sparc_profile_vo, -7, extra_weight=False)
plt.plot(sparc_profile_vo['time'], r_eff_vo['data'], '-o', ms=4, label='$R_{eff}$, Filled SPARC, ext weight False')
r_eff_vo = get_diag(sparc_profile_vo, -7, extra_weight=True)
plt.plot(sparc_profile_vo['time'], r_eff_vo['data'], '-o', ms=4, label='$R_{eff}$, Filled SPARC, ext weight True')
# belt
lat_ind = np.logical_and(sparc_profile_vo['lat'] >= -20, sparc_profile_vo['lat'] <= 20)
r_eff_vo = get_diag(sparc_profile_vo, -7, extra_weight=True, lat_limits=np.logical_not(lat_ind))
plt.plot(sparc_profile_vo['time'], r_eff_vo['data'], '-o', ms=4, label='$R_{eff}$, Filled SPARC, ext weight True, Lat [-20;20]')
plt.ylabel('Radius, ($\mu m$)')
plt.xlabel('Time, ()')
plt.legend()
save_figure(pics_output_folder, 'sparc asap r_eff')
plt.xlim(dt.datetime(1990,1,1), dt.datetime(1996,1,1))
save_figure(pics_output_folder, 'sparc asap r_eff 1990-1996')
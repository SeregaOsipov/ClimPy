import numpy as np
from scipy.io import loadmat

from climpy.utils.atmos_utils import air_number_density
from climpy.utils.file_path_utils import get_root_storage_path_on_hpc

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'


def get_scaling_settings_and_file_paths():
    """
    This is the list of Pinatubo scaling, precomputed in MATLAB, both RRTM & LBLRTM
    :return:
    """
    so2_ppmv_list = (0.109, 1.09, 10.9, 109)
    sulf_aod_list = (0.15, 1.09, 4.1, 16)

    # build list of the several volcano scales
    folder_path = get_root_storage_path_on_hpc() + '/Data/Papers/Toba/'
    model_v = 'rrtm'
    rrtm_file_paths_list = [folder_path + '{}/so2_h2so4_heating_rates_{}_sulf_od_{}.mat'.format(model_v, ppmv, aod) for
                            ppmv, aod in zip(so2_ppmv_list, sulf_aod_list)]
    model_v = 'rrtmg'
    rrtmg_file_paths_list = [folder_path + '{}/so2_h2so4_heating_rates_{}_sulf_od_{}.mat'.format(model_v, ppmv, aod) for
                            ppmv, aod in zip(so2_ppmv_list, sulf_aod_list)]

    n_vertical_layers = 50
    lblrtm_file_paths_list = [folder_path + '{}/so2_h2so4_j_diurnal_cycle_modelE_profile_so2_{}_sulf_od_{}_nLayers_{}.mat'.format('lblrtm', ppmv, aod, n_vertical_layers) for
                             ppmv, aod in zip(so2_ppmv_list, sulf_aod_list)]

    return so2_ppmv_list, sulf_aod_list, rrtm_file_paths_list, rrtmg_file_paths_list, lblrtm_file_paths_list


def prepare_matlab_rrtm_data(matlab_data_fp, extra_keys=[]):
    matlab_data = loadmat(matlab_data_fp)

    p_rho_grid = np.squeeze(matlab_data['swRrtmOutputVO_struct']['pressure'][0][0])
    # dimensions are: experiment (C, P), time, levels
    sw_hr = np.squeeze(matlab_data['swRrtmOutputVO_struct']['heatingRate'][0][0])
    lw_hr = np.squeeze(matlab_data['lwRrtmOutputVO_struct']['heatingRate'][0][0])

    sw_up_flux = np.squeeze(matlab_data['swRrtmOutputVO_struct']['upwardFlux'][0][0])
    sw_down_flux = np.squeeze(matlab_data['swRrtmOutputVO_struct']['downwardFlux'][0][0])
    # sw_net_flux = np.squeeze(matlab_data['swRrtmOutputVO_struct']['netFlux'][0][0])
    sw_net_flux = sw_down_flux-sw_up_flux

    lw_up_flux = np.squeeze(matlab_data['lwRrtmOutputVO_struct']['upwardFlux'][0][0])
    lw_down_flux = np.squeeze(matlab_data['lwRrtmOutputVO_struct']['downwardFlux'][0][0])
    # looks like lw net has wrong direction, derive it ourselfs
    # lw_net_flux = np.squeeze(matlab_data['lwRrtmOutputVO_struct']['netFlux'][0][0])
    lw_net_flux = lw_down_flux-lw_up_flux

    # sw values (heating rates and fluxes) have nan during the night, replace them with 0
    diags = (sw_hr, sw_up_flux, sw_down_flux, sw_net_flux)
    for diag in diags:
        ind = np.isnan(diag)
        diag[ind] = 0

    if np.sum(ind) == 0:
        print('REMEMBER TO CHECK continuous 24 hours to ensure the correctness of the DAILY MEAN averaging')
        print('MISSING night data, diurnal averaging will produce wrong results')
        #raise Exception('MISSING night data, diurnal averaging will produce wrong results')

    output_vo = {}

    output_vo['toa_index'] = 0
    output_vo['boa_index'] = -1

    output_vo['sw_hr'] = sw_hr
    output_vo['lw_hr'] = lw_hr

    output_vo['sw_up_flux'] = sw_up_flux
    output_vo['sw_down_flux'] = sw_down_flux
    output_vo['sw_net_flux'] = sw_net_flux

    output_vo['lw_up_flux'] = lw_up_flux
    output_vo['lw_down_flux'] = lw_down_flux
    output_vo['lw_net_flux'] = lw_net_flux

    # read the extra diags
    for key in extra_keys:
        output_vo[key] = np.squeeze(matlab_data[key])

    output_vo['p_rho'] = p_rho_grid
    # z grid has to be reversed to match p grid
    output_vo['z_stag'] = np.squeeze(matlab_data['z_stag'])

    return output_vo


def compute_rrtm_forcings(matlab_output_vo, p_index, c_index, agent_name, print_format='.2f'):

    toa_index = matlab_output_vo['toa_index']
    boa_index = matlab_output_vo['boa_index']

    # forcings
    sw_net_data = matlab_output_vo['sw_net_flux']
    lw_net_data = matlab_output_vo['lw_net_flux']

    sw_net_dm_data = np.mean(sw_net_data, axis=1)
    lw_net_dm_data = np.mean(lw_net_data, axis=1)

    sw_forcing_daily_mean = sw_net_dm_data[p_index]-sw_net_dm_data[c_index]
    lw_forcing_daily_mean = lw_net_dm_data[p_index]-lw_net_dm_data[c_index]

    print('daily mean (24h) forcings for ' + agent_name)
    string = 'TOA SW net : {:' + print_format + '} Wm^-2'
    print(string.format(sw_forcing_daily_mean[toa_index]))
    string = 'TOA LW net : {:' + print_format + '} Wm^-2'
    print(string.format(lw_forcing_daily_mean[toa_index]))
    string = 'TOA SW+LW net : {:' + print_format + '} Wm^-2'
    print(string.format(sw_forcing_daily_mean[toa_index]+lw_forcing_daily_mean[toa_index]))

    print('daily mean (24h) forcings for ' + agent_name)
    string = 'BOA SW net : {:' + print_format + '} Wm^-2'
    print(string.format(sw_forcing_daily_mean[boa_index]))
    string = 'BOA LW net : {:' + print_format + '} Wm^-2'
    print(string.format(lw_forcing_daily_mean[boa_index]))
    string = 'BOA SW+LW net : {:' + print_format + '} Wm^-2'
    print(string.format(sw_forcing_daily_mean[boa_index] + lw_forcing_daily_mean[boa_index]))

    return sw_forcing_daily_mean, lw_forcing_daily_mean


def prepare_matlab_disort_data(matlab_data_fp):
    matlab_data = loadmat(matlab_data_fp)

    # preprocessor
    if 'zStagGrid' in matlab_data.keys():
        matlab_data['z_stag'] = matlab_data['zStagGrid']
        matlab_data['z_stag_grid'] = matlab_data['zStagGrid']

    vo = {}
    vo['wl_grid'] = np.squeeze(matlab_data['waveLengthData']) #um

    keys = ('p_rho', 'p_stag', 't_stag', 'z_stag')
    for key in keys:
        if 'p_rho' in matlab_data.keys():
            vo[key] = np.squeeze(matlab_data[key])
        else:
            vo[key] = np.squeeze(matlab_data[key+'_grid'])

    # pp
    vo['z_rho_grid'] = (vo['z_stag'][1:] + vo['z_stag'][:-1])/2

    vo['o3_profile'] = np.squeeze(matlab_data['o3_ppmv_profile'])
    vo['so2_profile'] = np.squeeze(matlab_data['so2_ppmv_profile'])
    if 'h2so4_od_profile' in matlab_data.keys():
        vo['aer_od_profile'] = np.squeeze(matlab_data['h2so4_od_profile'])
    if 'aer_od_profile' in matlab_data.keys():
        vo['aer_od_profile'] = np.squeeze(matlab_data['aer_od_profile'])

    if 'spectralFluxDirDown' in matlab_data.keys():
        vo['spectral_actinic_flux'] = np.squeeze(matlab_data['spectralActinicFlux'])
        vo['spectral_flux_dir_down'] = np.squeeze(matlab_data['spectralFluxDirDown'])
        vo['spectral_flux_diff_down'] = np.squeeze(matlab_data['spectralFluxDiffDown'])
        vo['spectral_flux_up'] = np.squeeze(matlab_data['spectralFluxUp'])

    experiment_labels = ()
    if 'experimentLabels' in matlab_data.keys():
        for i in range(matlab_data['experimentLabels'].shape[1]):
            experiment_labels += (matlab_data['experimentLabels'][0][i][0],)

    vo['experiment_labels'] = experiment_labels

    # post processing
    dz = np.diff(vo['z_stag'])
    vo['aer_ext'] = vo['aer_od_profile'] / dz / 10 ** 3
    n_a = air_number_density(vo['p_stag'] * 10 ** 2, vo['t_stag'])
    vo['o3_nd'] = vo['o3_profile'] * 10 ** -6 * n_a

    return vo


def get_diag(vo, var_key, sim_index, anomaly_wrt_index=None, relative=False):  # applies to DISORT
    # matlab_output_vo dims: level, experiment, time,

    data_p = vo[var_key][:, sim_index, ]
    diag = data_p

    if anomaly_wrt_index is not None:
        data_c = vo[var_key][:, anomaly_wrt_index,]
        diag = data_p - data_c
        if relative:
            diag = (data_p - data_c) / data_c

    return diag


def get_rrtm_diag(matlab_output_vo, var_key, sim_index, anomaly_wrt_index=None):
    diags = matlab_output_vo[var_key]  # experiment, time, level

    diag = diags[sim_index]
    # compute anomaly (P-C)
    if anomaly_wrt_index is not None:
        diag = diags[sim_index] - diags[anomaly_wrt_index]

    # compute daily mean
    # diag_dm = np.mean(diags, axis=1)
    return diag

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import golden
from climpy.utils.matlab_data_utils import get_diag, get_rrtm_diag
from climpy.utils.plotting_utils import JGR_page_width_inches

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'


def plot_daily_mean_profile(matlab_output_vo, diag_name, p_index, c_index=None, y_fill_region=None):
    sw_hr_data = matlab_output_vo['sw_'+diag_name] #experiment, time, level
    lw_hr_data = matlab_output_vo['lw_'+diag_name]

    # compute daily mean
    sw_dm_data = np.mean(sw_hr_data, axis=1)
    lw_dm_data = np.mean(lw_hr_data, axis=1)

    # if c index is None, plot the sim_index
    # if anomaly_wrt_index is given, plot the anomaly
    sw_hr_data_to_plot = sw_dm_data[p_index]
    lw_hr_data_to_plot = lw_dm_data[p_index]

    # compute anomaly (P-C)
    if c_index is not None:
        sw_hr_data_to_plot = sw_dm_data[p_index] - sw_dm_data[c_index]
        lw_hr_data_to_plot = lw_dm_data[p_index] - lw_dm_data[c_index]

    p_data = matlab_output_vo['p_rho_grid']

    plt.figure()
    plt.grid(True)
    # first do the shading to indicate SO2 location
    if y_fill_region is not None:
        x_data = (-500, 500)
        plt.fill_between(x_data, y_fill_region[0], y_fill_region[1], facecolor='lightgray')
    #then do the HR
    plt.plot(sw_hr_data_to_plot, p_data, 'b', label='SW')
    plt.plot(lw_hr_data_to_plot, p_data, 'r', label='LW')
    plt.plot(sw_hr_data_to_plot+lw_hr_data_to_plot, p_data, 'k', label='NET')
    plt.yscale('log')
    plt.ylim((np.max(p_data), np.min(p_data)))

    plt.ylabel('pressure, (hPa)')
    plt.xlabel('heating rate, ($Kday^{-1}$)')

    plt.tick_params(right=True, which='both')

    plt.legend()
    plt.tight_layout()


def plot_j_profile(matlab_output_vo):
    vo = matlab_output_vo

    j_O3 = get_diag(vo, 'j_o3', 1, 0, True)
    j_O3 *= 10**2

    plt.figure(constrained_layout=True, figsize=(JGR_page_width_inches(), JGR_page_width_inches() / golden))
    plt.grid(True)

    # plot individual profiles
    plt.plot(j_O3, vo['p_stag_grid'], 'tab:blue', linewidth=0.1)
    # daily means
    plt.plot(np.mean(j_O3, axis=1), vo['p_stag_grid'], 'o-', color='tab:blue', ms=3, label='SW')

    plt.yscale('log')
    plt.ylim((np.max(vo['p_stag_grid']), np.min(vo['p_stag_grid'])))

    plt.ylabel('pressure, (hPa)')
    plt.xlabel('relative change, (%)')

    plt.tick_params(right=True, which='both')
    # plt.legend()


def plot_ozone_composite_diags(vo, sim_index, anomaly_wrt_index):
    fig, axes = plt.subplots(constrained_layout=True, nrows=1, ncols=2,
                             figsize=(JGR_page_width_inches(), JGR_page_width_inches() * 9 / 16))

    dz = np.diff(vo['z_stag_grid'])
    aer_ext_data = vo['aer_od_profile'] / dz / 10**3

    plt.sca(axes[0])
    plt.grid(True)

    plt.plot(aer_ext_data, vo['p_rho_grid'], label=r'Ash, $\tau(0.55 \mu m) $={:.1f}'.format(np.sum(vo['aer_od_profile'])))
    plt.yscale('log')
    plt.ylim((np.max(vo['p_rho_grid']), np.min(vo['p_rho_grid'])))
    plt.ylabel('Pressure, (hPa)')
    plt.xlabel('Extinction, ($km^{-1}$)')

    # axes[0].xaxis.set_major_formatter(ScalarFormatter(useOffset=True))
    # axes[0].xaxis.set_major_formatter(FixedScalarFormatter(-7))

    ax1 = axes[0]
    ax2 = ax1.twiny()
    # ax2.plot(sparc_profile_vo['so2_profile'], sparc_profile_vo['p_stag_grid'], '-r', label='$\mathrm{SO_{2}}$')
    ax2.plot(vo['o3_profile'], vo['p_stag_grid'], '-k', label='$\mathrm{O_{3}}$')
    ax2.set_ylim((np.max(vo['p_stag_grid']), np.min(vo['p_stag_grid'])))
    # ax2.set_xlabel('ppmv, ()')#, color='r')
    ax2.set_xlabel('Mixing ratio, (ppmv)')  # , color='r')
    # ax2.set_xlim((10**-5, 2*10**1))
    # ax2.set_xscale('log')
    # for t in ax2.get_xticklabels():
    #     t.set_color('r')
    # plt.legend(loc='upper right')

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles1 + handles2, labels1 + labels2, loc='upper right')

    plt.sca(axes[1])
    plt.grid(True)

    keys = ['j_o2_mean', 'j_o3_mean', 'actinic_flux_200_340_mean', 'actinic_flux_220_306_mean']
    labels = ['j(O$_2$)', 'j(O$_3$)', 'actinic f(200-340 nm)', 'actinic f(220-306 nm)']
    for j_var_key, legend_label in zip(keys, labels):
        rel_diff = get_diag(vo, j_var_key, sim_index, anomaly_wrt_index, True)
        # j_data_dm_P = np.mean(sparc_profile_vo[j_var_key][:, sim_index, :], axis=1)
        # j_data_dm_C = np.mean(sparc_profile_vo[j_var_key][:, anomaly_wrt_index, :], axis=1)
        # rel_diff = (j_data_dm_P - j_data_dm_C) / j_data_dm_C
        plt.plot(rel_diff, vo['p_stag_grid'], '-o', ms=2, label=legend_label)

    plt.yscale('log')
    plt.ylim((np.max(vo['p_stag_grid']), np.min(vo['p_stag_grid'])))
    # plt.ylabel('pressure, (hPa)')
    # plt.xlabel('j relative change, (dimensionless)')
    plt.xlabel('Relative change, ()')
    # plt.title('Photolysis rate change')  # , J, (P-C)/C
    # plt.legend(loc='upper left')
    plt.legend(loc='lower right')


def plot_o3_j_diags_publication(vo, sim_index, anomaly_wrt_index):
    # fig = plt.figure(constrained_layout=True, figsize=(JGR_page_width_inches() / 2, JGR_page_width_inches() / 2))
    plt.figure(constrained_layout=True, figsize=(JGR_page_width_inches()/2, JGR_page_width_inches() / golden))
    plt.grid(True)

    # plt.yscale('log')
    # plt.ylim((np.max(sparc_profile_vo['p_stag_grid']), np.min(sparc_profile_vo['p_stag_grid'])))
    # plt.ylabel('pressure, (hPa)')

    vert_coord = vo['z_stag'] * 10 ** -3  # km
    # plt.ylim((np.max(p_data), np.min(p_data)))
    plt.ylabel('Height, (km)')
    plt.xlabel('Relative change, (%)')

    keys = ['j_o3_mean', 'actinic_flux_200_340_mean']
    labels = ['O$_3$ photolysis rate', 'actinic flux, 200-340 nm']
    for j_var_key, legend_label in zip(keys, labels):
        rel_diff = get_diag(vo, j_var_key, sim_index, anomaly_wrt_index, True)
        plt.plot(rel_diff * 10 ** 2, vert_coord, '-o', ms=3, label=legend_label)

    # plt.title('Photolysis rate change')  # , J, (P-C)/C

    ax1 = plt.gca()
    ax2 = ax1.twiny()
    color = 'tab:cyan'
    ax2.plot(vo['o3_nd'] * 10**-18, vert_coord, '-', color=color, label='O$_{3}$ number density')
    # ax2.set_xlabel('ppmv, ()')#, color='r')
    ax2.set_xlabel('Mixing ratio, (ppmv)')  # , color='r')
    ax2.set_xlabel('Number density, ($10^{12}$ molecules cm$^{-3}$)', color=color)  # , color='r')
    plt.tick_params(axis='x', labelcolor=color)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    # ax2.legend(handles1 + handles2, labels1 + labels2, loc='lower right')
    ax1.legend(handles1, labels1, loc='lower right')
    ax2.legend(handles2, labels2, loc='upper left')


def plot_hr_profile(matlab_output_vo,sim_index, anomaly_wrt_index=None):
    sw_hr = get_rrtm_diag(matlab_output_vo, 'sw_hr', sim_index, anomaly_wrt_index)
    lw_hr = get_rrtm_diag(matlab_output_vo, 'lw_hr', sim_index, anomaly_wrt_index)

    # compute daily mean
    sw_hr_dm = np.mean(sw_hr, axis=0)
    lw_hr_dm = np.mean(lw_hr, axis=0)

    p_data = matlab_output_vo['p_rho']

    plt.figure(constrained_layout=True, figsize=(JGR_page_width_inches(), JGR_page_width_inches() / golden))
    plt.grid(True)

    # plot individual profiles
    plt.plot(sw_hr.transpose(), p_data, 'b', linewidth=0.1)
    plt.plot(lw_hr.transpose(), p_data, 'r', linewidth=0.1)
    # daily means
    plt.plot(sw_hr_dm, p_data, 'bo-', ms=3, label='SW')
    plt.plot(lw_hr_dm, p_data, 'r*-', ms=3, label='LW')
    plt.plot(sw_hr_dm+lw_hr_dm, p_data, 'k', label='SW+LW')

    plt.yscale('log')
    plt.ylim((np.max(p_data), np.min(p_data)))

    plt.ylabel('pressure, (hPa)')
    plt.xlabel('heating rate, ($Kday^{-1}$)')

    plt.tick_params(right=True, which='both')
    plt.legend()


def plot_hr_profile_publication(matlab_output_vo, sim_index, anomaly_wrt_index=None):
    sw_hr = get_rrtm_diag(matlab_output_vo, 'sw_hr', sim_index, anomaly_wrt_index)
    lw_hr = get_rrtm_diag(matlab_output_vo, 'lw_hr', sim_index, anomaly_wrt_index)

    # compute daily mean
    sw_hr_dm = np.mean(sw_hr, axis=0)
    lw_hr_dm = np.mean(lw_hr, axis=0)

    plt.figure(constrained_layout=True, figsize=(JGR_page_width_inches() / 2, JGR_page_width_inches() / 2))
    plt.grid(True)

    # pressure coord
    # vert_coord = matlab_output_vo['p_rho']  # hPa
    # plt.yscale('log')
    # plt.ylim((np.max(p_data), np.min(p_data)))
    # plt.ylabel('Pressure, (hPa)')

    # height coord
    vert_coord = matlab_output_vo['z_stag'] * 10**-3  # km
    # TODO: fix me, temp inverse z coord
    vert_coord = vert_coord[::-1]

    # plt.ylim((np.max(p_data), np.min(p_data)))
    plt.ylabel('Height, (km)')

    # plot individual profiles
    # plt.plot(sw_hr.transpose(), vert_coord, 'b', linewidth=0.1)
    # plt.plot(lw_hr.transpose(), vert_coord, 'r', linewidth=0.1)
    # daily means
    plt.plot(sw_hr_dm, vert_coord, 'b', ms=3, label='SW')
    plt.plot(lw_hr_dm, vert_coord, 'r', ms=3, label='LW')
    plt.plot(sw_hr_dm+lw_hr_dm, vert_coord, 'ko-', ms=3, label='SW+LW')

    plt.xlabel('Heating rate, (K day$^{-1}$)')

    # plt.tick_params(right=True, which='both')
    plt.legend()


def plot_fluxes_profile(matlab_output_vo, flux_direction, sim_index, anomaly_wrt_index=None):
    """

    :param flux_direction: up, down or net (down - up)
    :param sim_index:
    :param anomaly_wrt_index:
    :return:
    """
    sw_diag = get_rrtm_diag(matlab_output_vo, 'sw_{}_flux'.format(flux_direction), sim_index, anomaly_wrt_index)
    lw_diag = get_rrtm_diag(matlab_output_vo, 'lw_{}_flux'.format(flux_direction), sim_index, anomaly_wrt_index)

    # compute daily mean
    sw_diag_dm = np.mean(sw_diag, axis=0)
    lw_diag_dm = np.mean(lw_diag, axis=0)

    p_data = matlab_output_vo['p_rho']

    plt.figure(constrained_layout=True, figsize=(JGR_page_width_inches(), JGR_page_width_inches() / golden))
    plt.grid(True)

    # plot individual profiles
    plt.plot(sw_diag.transpose(), p_data, 'b', linewidth=0.1)
    plt.plot(lw_diag.transpose(), p_data, 'r', linewidth=0.1)
    # daily means
    plt.plot(sw_diag_dm, p_data, 'bo-', ms=3, label='SW')
    plt.plot(lw_diag_dm, p_data, 'r*-', ms=3, label='LW')
    plt.plot(sw_diag_dm + lw_diag_dm, p_data, 'k', label='SW+LW')

    plt.yscale('log')
    plt.ylim((np.max(p_data), np.min(p_data)))

    plt.ylabel('pressure, (hPa)')
    plt.xlabel('Flux, ($Wm^{-2}$)')

    plt.tick_params(right=True, which='both')
    plt.legend()


def plot_net_hr():
    plt.figure(constrained_layout=True, figsize=(JGR_page_width_inches()/2, JGR_page_width_inches() / golden))
    plt.cla()
    plt.grid(True)
    # plt.ylim((np.max(p_data), np.min(p_data)))
    plt.ylabel('Height, (km)')
    plt.xlabel('SW + LW heating rate, (K day$^{-1}$)')

    colors = ('k', 'tab:blue', 'k', 'k', 'k', 'tab:gray', 'tab:red', 'aqua')
    for i in (1, 6, 5, 7):
        sw_hr = get_rrtm_diag(matlab_output_vo, 'sw_hr', i, 0)
        lw_hr = get_rrtm_diag(matlab_output_vo, 'lw_hr', i, 0)

        # compute daily mean
        sw_hr_dm = np.mean(sw_hr, axis=0)
        lw_hr_dm = np.mean(lw_hr, axis=0)

        # height coord
        vert_coord = matlab_output_vo['z_stag'] * 10 ** -3  # km
        vert_coord = vert_coord[::-1]
        # plt.plot(sw_hr_dm, vert_coord, 'b', ms=3, label='SW')
        # plt.plot(lw_hr_dm, vert_coord, 'r', ms=3, label='LW')
        plt.plot(sw_hr_dm+lw_hr_dm, vert_coord, 'o-',
                 color=colors[i], ms=3, label='{}, ({} Mt)'.format(sim_labels[i], sim_total_injected_mass[i]))  # , SW+LW
                 # color = colors[i], ms = 3, label = '{}'.format(sim_labels[i]))  # , SW+LW

        # plt.tick_params(right=True, which='both')
        plt.legend()

    plt.ylim((5, 25))


def plot_input_profile(matlab_output_vo):
    p_stag = matlab_output_vo['p_stag']  # hPa
    z_stag = matlab_output_vo['z_stag'] / 10 ** 3  # km
    z_rho = (p_stag[:-1]+p_stag[1:])/2

    plt.figure(constrained_layout=True, figsize=(1.5 * JGR_page_width_inches(), 1.5 * JGR_page_width_inches() / golden))
    plt.grid(True)

    # plot individual profiles
    plt.plot(matlab_output_vo['q_profile_c_RH'], p_stag, '-*', label='q C, [RH, %]')
    plt.plot(matlab_output_vo['q_profile_p'], p_stag, '-o', label='q P, [gm^-3]')
    plt.plot(matlab_output_vo['q_profile_100rh'], p_stag, '-o', label='q 100% RH -> [gm^-3]')
    plt.plot(matlab_output_vo['q_profile_debug'], p_stag, '-1', label='q debug, [RH, %]')
    plt.plot(matlab_output_vo['so2_profile_p'], p_stag, '-s', label='SO2')
    plt.plot(matlab_output_vo['o3_profile_c'], p_stag, '-x', label='O_3 C, [ppmv]')
    plt.plot(matlab_output_vo['o3_profile_p'], p_stag, '-+', label='O_3 P, [ppmv]')
    wl = matlab_output_vo['sw_wavelengths'][3]

    keys = aerosol_keys[0:2]
    for aerosol_key in keys:
        aerosol_label = aerosol_key.split('_')[0]
        column_od = np.sum(matlab_output_vo[aerosol_key][4,:])
        plt.plot(matlab_output_vo[aerosol_key][4,:], z_rho, '-.', label='{} AOD@{:2.0f}nm, column is {:2.1f}'.format(aerosol_label, wl*1000, column_od))

    for cloud_key in cloud_keys:
        aerosol_label = cloud_key.split('_')[0]
        cwp = np.sum(matlab_output_vo[aerosol_key][4,:])
        plt.plot(matlab_output_vo[cloud_key][:], z_rho, '-.', label='{} CWP, column is {:2.1f} g/m^2'.format(aerosol_label, cwp))

    plt.xscale('log')
    plt.ylabel('Pressure, (hPa)')
    plt.xlabel('Rel Humidity, (%) or Mass density, ($gm^{-3}$) or AOD, ()')

    plt.tick_params(right=True, which='both')
    plt.legend()

    plt.title('Column model input profiles')

    p_ticks = p_stag[::5]
    z_ticks = z_stag[::5]
    # add km labels on the right
    ax1 = plt.gca()
    ax1.set_yscale('log')
    ax1.set_ylim((np.max(p_data), np.min(p_data)))
    ax1.set_yticks(p_ticks)
    ax1.set_ylabel('Pressure, (hPa)')
    ax1.yaxis.set_major_formatter(ticker.ScalarFormatter())

    ax2 = ax1.twinx()
    ax2.set_yscale('log')
    ax2.set_yticks(p_ticks)
    ax2.set_ylim((np.max(p_data), np.min(p_data)))
    ax2.yaxis.set_major_formatter(ticker.ScalarFormatter())

    z_tick_labels = ['{:.2f}'.format(z_tick) for z_tick in z_ticks]
    ax2.set_yticklabels(z_tick_labels)
    ax2.set_ylabel('Height, (km)')
    # ax1.get_yticks()


def plot_input_profile_publication(matlab_output_vo):
    z_stag = matlab_output_vo['z_stag'] / 10**3  # km
    z_rho = (z_stag[:-1]+z_stag[1:])/2
    dz = np.diff(z_stag)
    wl = matlab_output_vo['sw_wavelengths'][3]

    plt.figure(constrained_layout=True, figsize=(JGR_page_width_inches(), JGR_page_width_inches() / golden))
    plt.grid(True)

    # plot individual profiles
    plt.plot(matlab_output_vo['q_profile_p'], z_stag, 'b-', label='Water vapor concentration, ($gm^{-3}$)')

    keys = aerosol_keys[0:2]

    for aerosol_key, fc, ec, hatch in zip(keys, ['lightgray', 'none'], ['tab:gray', 'tab:red'], ['', 'o']):
        aerosol_label = aerosol_key.split('_')[0]
        column_od = np.sum(matlab_output_vo[aerosol_key][4,:])
        # plt.plot(matlab_output_vo[aerosol_key][4,:]/dz, z_rho, '-.', label='{} $\epsilon$ @ {:2.0f} nm, ($km^{{-1}}$); $\\tau$={:2.1f}'.format(aerosol_label.capitalize(), wl*1000, column_od))
        label = '{} extinction, ($km^{{-1}}$); $\\tau$({:0.1f} $\\mu$m)={:2.1f}'.format(aerosol_label.capitalize(), wl, column_od)
        plt.fill_betweenx(z_rho, matlab_output_vo[aerosol_key][4, :] / dz, '-.',
                          fc=fc, ec=ec, hatch=hatch, alpha=1.0, linewidth=1.0,
                        label=label)

    for cloud_key, ec, hatch in zip(cloud_keys, ['aqua',], ['/']):
        aerosol_label = cloud_key.split('_')[0]
        cwp = np.sum(matlab_output_vo[cloud_key][:])
        cloud_concentration = dz
        # plt.plot(matlab_output_vo[cloud_key][:]/dz, z_rho, '-.', label='{} water content, ($gm^{{-3}}$), WP={:2.1f}, ($gm^{{-2}}$)'.format(aerosol_label, cwp))
        # plt.plot(matlab_output_vo[cloud_key][:], z_rho, '-.', label='{} water path, ($gm^{{-2}}$), $\Sigma$={:2.1f}, ($gm^{{-2}}$)'.format(aerosol_label, cwp))
        plt.fill_betweenx(z_rho, matlab_output_vo[cloud_key][:]/dz*10**-3, '-.',
                         fc='none', ec=ec, hatch=hatch, alpha=1.0, linewidth=1.0,
                          label='{}, ($gm^{{-3}}$); IWP={:2.2f} $gm^{{-2}}$'.format(aerosol_label.capitalize(), cwp))
        # fill_betweenx(self, y, x1, x2=0, where=None, step=None, interpolate=False, *, data=None, **kwargs)[source]

    # plt.plot(matlab_output_vo['so2_profile_p'], z_stag, '-.', label='SO2, ($gm^{-3}$)')
    plt.fill_betweenx(z_stag, matlab_output_vo['so2_profile_p'], '-.',
                      fc='none', ec='tab:blue', hatch='.', alpha=1.0, linewidth=1.0,
                      label='SO$_2$, ($gm^{-3}$)')

    plt.xscale('log')
    # plt.ylim((np.max(p_data), np.min(p_data)))
    plt.ylim((-0.5, 35))

    plt.ylabel('Height, (km)')
    plt.xlabel('Mass concentration, ($gm^{{-3}}$) or Aerosols extinction, ($km^{-1}$)')

    # plt.legend(loc='upper right')

    ax1 = plt.gca()
    ax2 = ax1.twiny()
    plt.sca(ax2)
    color = 'tab:brown'
    plt.plot(matlab_output_vo['t_stag'], z_stag, '-', color=color, label='Temperature, (K)')
    plt.xlabel('Temperature, (K)', color=color)
    plt.tick_params(axis='x', labelcolor=color)

    # plt.legend(loc='upper left')

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    # ax2.legend(handles2 + handles1, labels2 + labels1, loc='upper right')
    ax1.legend(handles1, labels1, loc='lower left')
    ax2.legend(handles2, labels2, loc='upper right')

    # plt.tick_params(right=True, which='both')
    # plt.title('Input profiles for the column model')
    return ax1, ax2


def plot_q_profiles_publication():
    z_stag = matlab_output_vo['z_stag'] / 10**3  # km
    z_rho = (z_stag[:-1]+z_stag[1:])/2
    dz = np.diff(z_stag)
    wl = matlab_output_vo['aerosols_wls'][3]

    plt.figure(constrained_layout=True, figsize=(JGR_page_width_inches(), JGR_page_width_inches() / golden))
    plt.grid(True)

    # plot individual profiles
    plt.plot(matlab_output_vo['q_profile_100rh'], z_stag, '-', label='100% RH -> q, ($gm^{-3}$)')
    # plt.plot(matlab_output_vo['q_profile_p'], z_stag, '-', label='q P, ($gm^{-3}$)')
    plt.plot(matlab_output_vo['q_profile_c'], z_stag, '-', label='q, ($gm^{-3}$)')
    # plt.plot(matlab_output_vo['q_profile_p_RH'], z_stag, '-', label='q P, ($gm^{-3}$)')

    # plt.fill_betweenx(z_rho, 0, 100, where=np.logical_and(z_rho >= 15, z_rho <= 19), facecolor='lightgray', alpha=0.5)
    ax = plt.gca()
    plt.fill_betweenx(z_rho, 0, 1, where=np.logical_and(z_rho >= 15, z_rho <= 19),
                      label='Volcanic plume, 17 km',
                      facecolor='darkgray', ec='gray', alpha=0.5, transform=ax.get_yaxis_transform())

    plt.fill_betweenx(z_rho, 0, 1, where=np.logical_and(z_rho >= 22, z_rho <= 26),
                      label='Volcanic plume, 24 km',
                      facecolor='moccasin', ec='orange', alpha=0.5, transform=ax.get_yaxis_transform())

    plt.xscale('log')
    plt.ylim((-0.5, 35))

    plt.ylabel('Height, (km)')
    plt.xlabel('Mass concentration, ($gm^{{-3}}$)')

    # plt.legend(loc='lower left')

    ax1 = plt.gca()
    ax2 = ax1.twiny()
    plt.sca(ax2)
    plt.plot(matlab_output_vo['q_profile_c_RH'], z_stag, '-', color='tab:red', label='RH, (%)')
    # plt.legend(loc='upper left')
    plt.xlabel('Relative humidity, (%)')

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    plt.legend(h1 + h2, l1 + l2)


def plot_theta_profile_diags():
    z_stag = matlab_output_vo['z_stag'] / 10**3  # km
    p_stag = matlab_output_vo['p_stag']  # hPa
    z_rho = (z_stag[:-1]+z_stag[1:])/2

    plt.figure(constrained_layout=True, figsize=(JGR_page_width_inches(), JGR_page_width_inches() / golden))
    plt.grid(True)

    dThetaDz = np.gradient(matlab_output_vo['theta_stag'], z_stag)
    plt.plot(dThetaDz, p_stag, '-o', label='$d\Theta/dz$')
    plt.legend()

    # plot individual profiles
    # plt.plot(matlab_output_vo['theta_stag'], z_stag, '-*', label='$\Theta$')
    # plt.plot(dThetaDz, z_stag, '-o', label='$d\Theta/dz$')

    plt.yscale('log')
    plt.ylim((np.max(p_stag), np.min(p_stag)))
    plt.xlabel('Temperature gradient, (K/km)')
    plt.ylabel('Pressure, (hPa)')
    plt.tick_params(right=True, which='both')
    ind = np.logical_and(p_stag >= 10, p_stag<=100)
    mean_value = np.mean(dThetaDz[ind])
    plt.title('Potential temperature gradient $\\nabla\Theta$ \n 100-10 hPa mean is {:.1f} K/km'.format(mean_value))

    # plt.xscale('log')
    # plt.ylabel('Height, (km)')
    # plt.xlabel('Rel Humidity, (%) or Mass density, ($gm^{-3}$) or AOD, ()')
    #
    # plt.tick_params(right=True, which='both')
    # plt.legend()
    #
    # plt.title('Column model input profiles')

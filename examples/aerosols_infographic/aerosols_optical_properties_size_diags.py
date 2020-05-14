import climpy.utils.mie_utils as mie
import climpy.utils.aeronet_utils as aeronet
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import os
from climpy.utils.plotting_utils import get_JGR_full_page_width_inches, save_figure_bundle

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

"""
https://github.com/SeregaOsipov/ClimPy/wiki/Aerosols-infographic
"""

station = 'KAUST'
# CASE 1: fine mode volume = coarse
time_range = [dt.datetime(2012, 9, 28, 0, 0), dt.datetime(2012, 9, 29, 0, 0)]
# CASE 2: coarse mode dominates = coarse
time_range = [dt.datetime(2012, 9, 29, 0, 0), dt.datetime(2012, 9, 30, 0, 0)]
res = aeronet.DAILY
# get Aeronet refrative index
ri_vo = aeronet.get_refractive_index('*{}*'.format(station), level=15, res=res, time_range=time_range)
# and size distribution
sd_vo = aeronet.get_size_distribution('*{}*'.format(station), level=15, res=res, time_range=time_range)

# and AOD to verify Mie calculations
aod443_vo = aeronet.get_aod_diag('*{}*'.format(station), 'AOD_443nm', level=15, res=res, time_range=time_range)
aod532_vo = aeronet.get_aod_diag('*{}*'.format(station), 'AOD_532nm', level=15, res=res, time_range=time_range)
aod667_vo = aeronet.get_aod_diag('*{}*'.format(station), 'AOD_667nm', level=15, res=res, time_range=time_range)

# drop the time dimension (only 1 month due to time range)
time_ind = 0
ri_vo['data'] = ri_vo['data'][time_ind]
sd_vo['data'] = sd_vo['data'][time_ind]
aod443_vo['data'] = aod443_vo['data'][time_ind]
aod532_vo['data'] = aod532_vo['data'][time_ind]
aod667_vo['data'] = aod667_vo['data'][time_ind]

# Compute AOD, first Mie extinction coefficients
r_data = np.logspace(-3, 2, 100)  # um
r_data = sd_vo['radii']  # actually use Aeronet reported radii
mie_vo = mie.get_mie_efficiencies(ri_vo['data'], r_data, ri_vo['wl']/10**3)

wl_index = 0
cross_section_area_transform = 3/4 * r_data**-1
od = np.trapz(mie_vo['qext'] * sd_vo['data'] * cross_section_area_transform, np.log(r_data), axis=1)


def get_cdf(f, radii):
    """
    Compute cdf: integral of f(x)*dx up to x
    :param f:
    :param radii:
    :return:
    """
    cdf = []
    for r in radii:
        ind = radii <= r
        cdf.append(np.trapz(f[ind], radii[ind]))
    return np.array(cdf)


# compute the CDFs for volume/mass and AOD
volume_cdf = get_cdf(sd_vo['data'], np.log(r_data))
area_cdf = get_cdf(sd_vo['data'] * cross_section_area_transform, np.log(r_data))
aod_cdf = get_cdf(mie_vo['qext'][0] * sd_vo['data'] * cross_section_area_transform, np.log(r_data))


# DO THE PLOTTING
fig = plt.figure(constrained_layout=True, figsize=(get_JGR_full_page_width_inches()*1.15, get_JGR_full_page_width_inches()))
# fig = plt.figure(figsize=(get_JGR_full_page_width_inches()*1., get_JGR_full_page_width_inches()))
gs = fig.add_gridspec(ncols=2, nrows=4, width_ratios=[1, 3], height_ratios=[3, 10, 10, 10])

# HEADER
ax_text = fig.add_subplot(gs[0, :])
ax_text.annotate(r'Aerosols: mass vs optical depth', (0.5, 0.5),
                 fontsize='xx-large',
                 xycoords='axes fraction', va='center', ha='center')
ax_text.axis('off')

ax_text = fig.add_subplot(gs[1, 0])
ax_text.annotate('Area under the curves\n\nrepresents volume (mass) and \ncross section (attenuation of light)',
                 (0.5, 0.5),
                 xycoords='axes fraction', va='center', ha='center')
ax_text.axis('off')
#r'\textcolor{red}{Today} '+

ax_text = fig.add_subplot(gs[2, 0])
ax_text.annotate('Extinction efficiency of each particle.\n\n$\lambda$ = 440 nm', (0.5, 0.5),
                 xycoords='axes fraction', va='center', ha='center')
ax_text.axis('off')

ax_text = fig.add_subplot(gs[3, 0])
ax_text.annotate('Contribution of particles that are <= r.\n\nNormalized to 1.', (0.5, 0.5),
                 xycoords='axes fraction', va='center', ha='center')
ax_text.axis('off')

# the plots itself
ax = fig.add_subplot(gs[1, 1])
plt.sca(ax)
plt.plot(sd_vo['radii'], sd_vo['data'], '-o', label='Volume')
plt.xscale('log')
# plt.xlabel('Radius, ($\mu m$)')
plt.ylabel('dV/dlnr [$\mu m^3$ $\mu m^{-2}$]')
plt.title('Size distributions')
# plt.legend(loc='upper left')

color = 'tab:orange'
ax2 = ax.twinx()
ax2.tick_params(axis='y', labelcolor=color)
plt.plot(sd_vo['radii'], sd_vo['data']*3/4*r_data**-1, '-o', color=color, label='Area')
plt.ylabel('dA/dlnr [$\mu m^2$ $\mu m^{-2}$]', color=color)
# plt.legend(loc='upper right')

h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax2.legend(h1+h2, l1+l2, loc='upper right')

# ax = fig.add_subplot(gs[2, 1])
# plt.sca(ax)
# plt.plot(ri_vo['wl'], np.real(ri_vo['data']), '-o')
# plt.xlabel('Wavelength, ($\mu m$)')
# plt.ylabel('Re(refractive index)')
#
# color = 'tab:gray'
# ax2 = ax.twinx()
# ax2.tick_params(axis='y', labelcolor=color)
# plt.plot(ri_vo['wl'], np.imag(ri_vo['data']), '-o', color=color)
# plt.ylabel('Im(refractive index)', color=color)
# plt.title('Refractive index')

ax = fig.add_subplot(gs[2, 1])
plt.sca(ax)
plt.plot(mie_vo['r_data'], mie_vo['qext'][wl_index], '-o')
plt.xscale('log')
# plt.xlabel('Radius, ($\mu m$)')
plt.ylabel('Extinction coefficient, ()')
plt.title('Mie $Q_{ext}$')

ax = fig.add_subplot(gs[3, 1])
plt.sca(ax)
plt.grid()
plt.plot(sd_vo['radii'], volume_cdf/volume_cdf[-1], '-o', label='Volume')
plt.plot(sd_vo['radii'], area_cdf/area_cdf[-1], '-o', label='Area')
plt.plot(sd_vo['radii'], aod_cdf/aod_cdf[-1], '-o', label='AOD')
plt.xscale('log')
plt.xlabel('Radius, ($\mu m$)')
plt.ylabel('CDF, ()')
plt.title('Cumulative distribution functions')  #  \n normalized to 1
plt.legend()

# text at the bottom
date_str = time_range[0].strftime('%Y-%m-%d')
plt.annotate('Aeronet station @ {} on {}'.format(station, date_str),
             (0.99, 0.015), fontsize='x-small', xycoords='figure fraction', va='center', ha='right')
url = 'https://github.com/SeregaOsipov/ClimPy/wiki/Aerosols-infographic'
# have to create new axis because of the bug in constrained_layout
ax_b = plt.axes((0.01, 0.015, 0.5, 0.05), facecolor='w')
ax_b.annotate('Sergey Osipov. Source {}'.format(url),
             (0.01, 0.015), fontsize='x-small', xycoords='figure fraction', va='center', ha='left')
ax_b.axis('off')
# plt.tight_layout()
save_figure_bundle(os.path.expanduser('~') + '/Pictures/Papers/infographics/aerosols/', 'Aerosols size distribution and optical properties {}'.format(date_str))

import climpy.utils.mie_utils as mie
import climpy.utils.aeronet_utils as aeronet
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import os
from climpy.utils.plotting_utils import JGR_page_width_inches, save_figure_bundle
from climpy.utils.refractive_index_utils import get_Williams_Palmer_refractive_index
from climpy.utils.stats_utils import get_cdf

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

"""
Derived from the https://github.com/SeregaOsipov/ClimPy/wiki/Aerosols-infographic

Here we use the size distribution of the volcanic aerosols to show the contribution to the mass and optical depth
We consider 2 SD cases, before Pinatubo and after. The effective radius is taken from SPARC ASAP.

The main purpose of the diagnostics is to show that background r_eff is too large is SPARC ASAP 
"""

sd_ind = 0
sd_ind = 1
sd_ind = 2

dp = np.logspace(-9, -4, 40)
dp = np.logspace(-8, -5, 40)
r_data = dp/2  # m
wavelengths = np.array([0.525, 1.02])  # um

# Prepare RI
ri_vo = get_Williams_Palmer_refractive_index(wavelengths=wavelengths)

# Compute Mie extinction coefficients
mie_vo = mie.get_mie_efficiencies(ri_vo['ri'], r_data*10**6, ri_vo['wl'])

# sample the SD
# sg, dg, moment3, moment0
r_eff = np.array([0.1, 0.3, 0.55])  # um. Background, sparc bg, max after Pinatubo
sg = np.ones(r_eff.shape) * 1.8  # fix the parameter
# r_e = r_g * exp(5/2 ln(sg)^2)
dg = 2 * r_eff / np.exp(5/2*np.log(sg)**2)
dg *= 10**-6  # um -> m

# sample the SD, normalized to 1
# THIS one is faster then sp.stats.lognorm
dNdlogp = 1/((2*np.pi)**(1/2) * np.log(sg[..., np.newaxis])) * np.exp(-1/2 * (np.log(dp)-np.log(dg[..., np.newaxis]))**2 / np.log(sg[..., np.newaxis])**2)

sd_vo = {}
sd_vo['data'] = dNdlogp[sd_ind]
sd_vo['radii'] = r_data

wl_index = 1
cross_section_area_transform = np.pi * r_data**2
od = np.trapz(mie_vo['qext'] * sd_vo['data'] * cross_section_area_transform, np.log(r_data), axis=1)

# compute the CDFs for volume/mass and AOD
volume_cdf = get_cdf(sd_vo['data'], np.log(r_data))
area_cdf = get_cdf(sd_vo['data'] * cross_section_area_transform, np.log(r_data))
aod_cdf = get_cdf(mie_vo['qext'][wl_index] * sd_vo['data'] * cross_section_area_transform, np.log(r_data))


# DO THE PLOTTING


fig = plt.figure(constrained_layout=True, figsize=(JGR_page_width_inches() * 1.15, JGR_page_width_inches()))
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
ax_text.annotate('Extinction efficiency of each particle.\n\n$\lambda$ = {} nm'.format(wavelengths[wl_index]*10**3), (0.5, 0.5),
                 xycoords='axes fraction', va='center', ha='center')
ax_text.axis('off')

ax_text = fig.add_subplot(gs[3, 0])
ax_text.annotate('Contribution of particles that are <= r.\n\nNormalized to 1.', (0.5, 0.5),
                 xycoords='axes fraction', va='center', ha='center')
ax_text.axis('off')

# the plots itself
ax = fig.add_subplot(gs[1, 1])
plt.sca(ax)
plt.plot(sd_vo['radii']*10**6, sd_vo['data']*4/3*np.pi*sd_vo['radii']**3, '-o', label='Volume')
plt.xscale('log')
# plt.xlabel('Radius, ($\mu m$)')
plt.ylabel('dV/dlnr [$\mu m^3$ $\mu m^{-2}$]')
plt.title('Size distributions')
# plt.legend(loc='upper left')

color = 'tab:orange'
ax2 = ax.twinx()
ax2.tick_params(axis='y', labelcolor=color)
plt.plot(sd_vo['radii']*10**6, sd_vo['data']*np.pi*sd_vo['radii']**2, '-o', color=color, label='Area')
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
plt.plot(sd_vo['radii']*10**6, volume_cdf/volume_cdf[-1], '-o', label='Volume')
plt.plot(sd_vo['radii']*10**6, area_cdf/area_cdf[-1], '-o', label='Area')
plt.plot(sd_vo['radii']*10**6, aod_cdf/aod_cdf[-1], '-o', label='AOD')
plt.xscale('log')
plt.xlabel('Radius, ($\mu m$)')
plt.ylabel('CDF, ()')
plt.title('Cumulative distribution functions')  #  \n normalized to 1
plt.legend()

# text at the bottom
# date_str = time_range[0].strftime('%Y-%m-%d')
plt.annotate('r_eff is {} $\mu m$, sigma is {}'.format(r_eff[sd_ind], sg[sd_ind]),
             (0.99, 0.015), fontsize='x-small', xycoords='figure fraction', va='center', ha='right')
url = 'https://github.com/SeregaOsipov/ClimPy/wiki/Aerosols-infographic'
# have to create new axis because of the bug in constrained_layout
ax_b = plt.axes((0.01, 0.015, 0.5, 0.05), facecolor='w')
ax_b.annotate('Sergey Osipov. Source {}'.format(url),
             (0.01, 0.015), fontsize='x-small', xycoords='figure fraction', va='center', ha='left')
ax_b.axis('off')
# plt.tight_layout()
save_figure_bundle(os.path.expanduser('~') + '/Pictures/Papers/infographics/aerosols/sparc asap', 'Volcanic sulfate size distribution and optical properties, r_eff is {}'.format(r_eff[sd_ind]))




# test how much more mass is needed to get the same OD at 1 um

sd_vo = {}
sd_vo['data'] = dNdlogp
sd_vo['radii'] = r_data

wl_index = 1
cross_section_area_transform = np.pi * r_data**2
od = np.trapz(mie_vo['qext'][wl_index] * sd_vo['data'] * cross_section_area_transform, np.log(r_data), axis=1)

m0_scale = od[1]/od[0]
print('To match the OD with r_eff 0.1 we need {} more particles (m0) compared to r_eff 0.3'.format(m0_scale))
m3s = np.trapz(sd_vo['data'] * 4/3*np.pi*sd_vo['radii']**3, np.log(r_data), axis=1)
print('Which is equivalent to {} more volume/mass (m3)'.format(m0_scale * m3s[1]/m3s[0]))

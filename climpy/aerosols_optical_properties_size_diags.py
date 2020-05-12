import climpy.utils.mie_utils as mie
import climpy.utils.aeronet_utils as aeronet
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from climpy.utils.plotting_utils import get_JGR_full_page_width_inches

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

station = 'KAUST'
# CASE 1: fine mode volume = coarse
time_range = [dt.datetime(2012, 9, 28, 0, 0), dt.datetime(2012, 9, 29, 0, 0)]
# CASE 2: coarse mode dominates = coarse
# time_range = [dt.datetime(2012, 9, 29, 0, 0), dt.datetime(2012, 9, 30, 0, 0)]
res = aeronet.DAILY
# get Aeronet refrative index
ri_vo = aeronet.get_refractive_index('*{}*'.format(station), level=15, res=res, time_range=time_range)
# and size distribution
sd_vo = aeronet.get_size_distribution('*{}*'.format(station), level=15, res=res, time_range=time_range)

# and AOD to verify Mie calculations
aod532_vo = aeronet.get_aod_diag('*{}*'.format(station), 'AOD_532nm', level=15, res=res, time_range=time_range)
aod443_vo = aeronet.get_aod_diag('*{}*'.format(station), 'AOD_443nm', level=15, res=res, time_range=time_range)

# drop the time dimension (only 1 month due to time range)
time_ind = 0
ri_vo['data'] = ri_vo['data'][time_ind]
sd_vo['data'] = sd_vo['data'][time_ind]
aod532_vo['data'] = aod532_vo['data'][time_ind]
aod443_vo['data'] = aod443_vo['data'][time_ind]

# Compute AOD, first Mie extinction coefficients
r_data = np.logspace(-3, 2, 100)  # um
r_data = sd_vo['radii']  # actually use Aeronet reported radii
mie_vo = mie.get_mie_efficiencies(ri_vo['data'], r_data, ri_vo['wl']/10**3)

wl_index = 0
cross_section_area_transform = r_data**-1
od = np.trapz(mie_vo['qext'] * sd_vo['data'] * cross_section_area_transform, r_data, axis=1)


fig, axes = plt.subplots(nrows=2, ncols=2, constrained_layout=True,
             figsize=(2*get_JGR_full_page_width_inches(), get_JGR_full_page_width_inches()))
plt.sca(axes.flatten()[0])
plt.plot(sd_vo['radii'], sd_vo['data'], '-o')
plt.xscale('log')
plt.xlabel('Radius, ($\mu m$)')
plt.ylabel('dV/dlnr [$\mu m^3$ $\mu m^{-2}$]')
plt.title('Volume size distribution')

ax = axes.flatten()[1]
plt.sca(ax)
plt.plot(ri_vo['wl'], np.real(ri_vo['data']), '-o')
plt.xlabel('Wavelength, ($\mu m$)')
plt.ylabel('Re(refractive index)')
ax2 = ax.twinx()
plt.plot(ri_vo['wl'], np.imag(ri_vo['data']), '-o', color='gray')
plt.ylabel('Im(refractive index)')
plt.title('Refractive index')

plt.sca(axes.flatten()[2])
plt.plot(mie_vo['r_data'], mie_vo['qext'][wl_index], '-o')
plt.xscale('log')
plt.xlabel('Radius, ($\mu m$)')
plt.ylabel('Extinction coefficient, ()')
plt.title('Mie qext')
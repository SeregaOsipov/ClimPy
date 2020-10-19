from climpy.utils.file_path_utils import get_pictures_root_folder
from climpy.utils.sparc_asap_sato_cmip_utils import prepare_sparc_asap_stratospheric_optical_depth, \
    prepare_sparc_asap_profile_data, disassemble_sparc_into_diags, derive_sparc_so4_wet_mass
from climpy.utils.plotting_utils import save_figure
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

'''
This script derived the mass of the volcanic aerosols for the SPARC ASAP data set.
I first restore the size distribution from the effective radius and then derive
the number of particles (0th moment) to match the extinction.
Then I compute the 3rd moment for a given effective radius.

See https://www.sparc-climate.org/wp-content/uploads/sites/5/2017/12/SPARC_Report_No4_Feb2006_ASAP.pdf 
'''

pics_output_folder = get_pictures_root_folder() + '/Papers/PinatuboEmac/SPARC_ASAP/'

sparc_od_vo = prepare_sparc_asap_stratospheric_optical_depth()
sparc_profile_vo = prepare_sparc_asap_profile_data(True, filter_unphysical_data=True)
r_eff_vo, ext_1020_vo, aod_1020_vo = disassemble_sparc_into_diags(sparc_profile_vo)

so4_mass_vo = derive_sparc_so4_wet_mass(r_eff_vo, ext_1020_vo)

# Check the Total m0

time_ind = 80
x_coord = sparc_profile_vo['time']
data_to_plot = np.nansum(moment0, axis=(1, 2))
plt.ion()
plt.figure()
plt.clf()
plt.plot(x_coord, data_to_plot, '-o')
plt.yscale('log')
# plt.ylim((1.e11, 1.e50))
# plt.xlim((dt.datetime(1990, 1,1), dt.datetime(1995, 1,1)))
plt.ylabel('SO$_{4}$ m0, (#)')
plt.xlabel('Time, ()')

# Check the Total Mass

x_coord = sparc_profile_vo['time']
data_to_plot = np.nansum(so4_mass_vo['data'], axis=(1, 2))
plt.ion()
plt.figure()
plt.clf()
plt.plot(x_coord, data_to_plot, '-o')
plt.yscale('log')
# plt.ylim((1.e-7, 1.e30))
plt.ylabel('SO$_{4}$ mass, (kg)')
plt.xlabel('Time, ()')
plt.title('Mass derived from SPARC ASAP using Mie')

save_figure(pics_output_folder, 'so4 mass')


# check what is wrong at ti=115

plt.figure()
plt.clf()
plt.contourf(sparc_profile_vo['altitude'], sparc_profile_vo['lat'], moment0[115])
plt.colorbar()
# find the location of the artifact

time_ind = np.nanargmax(data_to_plot)
x_coord[time_ind]
ind = np.unravel_index(np.nanargmax(moment0[time_ind], axis=None), moment0[time_ind].shape)  # (26, 49)

moment0[time_ind,...]
moment0[time_ind, 17, 57]
ext[time_ind, ..., 1]
ext[time_ind, 17, 57, 1]
dNdlogr[time_ind,..., 1]
dNdlogr[time_ind,19,59, 1]
r_eff[time_ind,...]
r_eff[time_ind,17, 57]
ext_1020[time_ind,...]
ext_1020[time_ind,17, 57]


ind = np.unravel_index(np.nanargmax(r_eff, axis=None), r_eff.shape)  # (230, 26, 49)
r_eff[230, 26, 49]
r_eff[229, 26, 49]



sparc_profile_vo['time'][time_ind]


# Mass plot


plt.figure()
plt.clf()
plt.contourf(sparc_profile_vo['altitude'], sparc_profile_vo['time'], mass[:, 16, :])
plt.colorbar()
plt.xlabel('Height, (km)')
plt.ylabel('Time, ()')
plt.ylim((dt.datetime(1990,1,1), dt.datetime(1994, 1, 1)))
plt.title('Mass, (kg)')

# R_eff plot

plt.figure()
plt.clf()
plt.contourf(sparc_profile_vo['altitude'], sparc_profile_vo['time'], r_eff[:, 16, :])
plt.colorbar()
plt.xlabel('Height, (km)')
plt.ylabel('Time, ()')
plt.ylim((dt.datetime(1990,1,1), dt.datetime(1994, 1, 1)))
plt.title('r_eff, ($\mu m$)')

# cell volume

plt.figure()
plt.clf()
plt.contourf(sparc_profile_vo['altitude'], sparc_profile_vo['lat'], cell_volume)
plt.colorbar()
plt.xlabel('Height, (km)')
plt.ylabel('Time, ()')
plt.title('r_eff, ($\mu m$)')



plt.ion()
plt.figure()
plt.clf()
plt.contourf(sparc_profile_vo['altitude'], sparc_profile_vo['lat'], mass[time_ind])
plt.colorbar()

plt.clf()
plt.contourf(sparc_profile_vo['altitude'], sparc_profile_vo['lat'], ext_1020[time_ind])
plt.colorbar()

plt.clf()
plt.contourf(sparc_od_vo['lat'], sparc_od_vo['time'], sparc_od_vo['data'])
plt.colorbar()



plt.ion()
plt.figure()
plt.cla()
plt.plot(dp, dNdlogp[100, 10, 10, :])
plt.plot(dp, dNdlogp[100, 10, 10, :]*cross_section_area_transform)
plt.xscale('log')
plt.yscale('log')




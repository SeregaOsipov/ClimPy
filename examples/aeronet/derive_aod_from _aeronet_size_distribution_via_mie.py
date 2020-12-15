import datetime as dt
import climpy.utils.aeronet_utils as aeronet
import matplotlib.pyplot as plt

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

station='KAUST_Campus'
aeronet_lvl = 15
level=aeronet_lvl
res=aeronet.ALL_POINTS
time_range = [dt.datetime(2012, 9, 29, 0, 0), dt.datetime(2012, 9, 30, 0, 0)]
time_range = [dt.datetime(2012, 9, 23, 0, 0), dt.datetime(2012, 9, 30, 0, 0)]

# derive Aeronet AOD vie Mie from dVdlnr
aeronet_pp_od_df = aeronet.derive_aod_from_size_distribution_using_mie(station, level=aeronet_lvl, res=res, time_range=time_range)
# rename columns
aeronet_pp_od_df.columns = ['Mie AOD at {} $\mu m$'.format(column) for column in aeronet_pp_od_df.columns]

# get Aeronet obserations of the AOD
aeronet_aod_df = aeronet.get_aod_product('*{}*'.format(station), level=aeronet_lvl, res=res, time_range=time_range)
# subset few wavelengths
aeronet_aod_df_subset = aeronet_aod_df[['AOD_532nm', 'AOD_1020nm']]



# plot the comparison
plt.figure(constrained_layout=True, figsize=(16, 9))
ax = plt.gca()

plt.cla()
aeronet_aod_df_subset.plot(ax=ax, style='o')
# plt.xlim(time_range[0], time_range[1])
aeronet_pp_od_df.plot(ax=ax, style='*')

plt.title('Aeronet observed AOD vs AOD derivied using dVdlnr and Mie\nDust RI is assumed for entire column dVdlnr')
plt.xlabel('Time, ()')
plt.ylabel('Optical depth, ()')

from climpy.utils.file_path_utils import get_pictures_root_folder
from climpy.utils.plotting_utils import save_figure
pics_output_folder = get_pictures_root_folder() + '/Papers/AirQuality/AQABA' + '/aod_diags/'
save_figure(pics_output_folder, 'Aeronet observed and Mie AOD comparison')
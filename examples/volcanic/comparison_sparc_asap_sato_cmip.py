from climpy.utils.diag_decorators import lonlat_weight_averaging
from climpy.utils.file_path_utils import get_pictures_root_folder
from climpy.utils.sparc_asap_sato_cmip_utils import prepare_sparc_asap_stratospheric_optical_depth, \
    prepare_sparc_asap_profile_data
from climpy.utils.avhrr_utils import prepare_avhrr_aod
from climpy.utils.climatology_utils import compute_daily_climatology
from climpy.utils.plotting_utils import save_figure, save_figure_bundle
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from climpy.TimeRangeVO import TimeRangeVO

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

pics_output_folder = get_pictures_root_folder() + '/Papers/PinatuboInitialStage/volcanic_data_sets_comparison/'

sparc_od_vo = prepare_sparc_asap_stratospheric_optical_depth()
sparc_profile_filled_vo = prepare_sparc_asap_profile_data(True)
# sparc_profile_output_set_unfilled = prepare_sparc_asap_vertically_resolved_data(None, False)

climatology_time_range_vo = TimeRangeVO(dt.datetime(1989, 1, 1), dt.datetime(1991, 1, 1))
avhrr_vo = prepare_avhrr_aod(time_range=(dt.datetime(1990, 1, 1), dt.datetime(1992, 1, 1)), zonal_mean=True)

anomaly_data, clim_data, yearly_data, unique_years = compute_daily_climatology(avhrr_vo['data'], avhrr_vo['time'], climatology_time_range_vo=climatology_time_range_vo)

# Plot the AVHRR AOD
plt.figure()
plt.contourf(avhrr_vo['data'], np.linspace(0, 0.5, 21), extend='both')
plt.colorbar()

# and the anomaly
plt.figure()
plt.contourf(anomaly_data, np.linspace(0, 0.5, 21), extend='both')
plt.colorbar()


# compare the two datasets
plt.figure(constrained_layout=True, figsize=(10, 5))
# plt.grid(True)


def get_diag(vo):
    ind = np.isnan(vo['data'])
    print('{} NaN values of the column AOD where ignored'.format(np.sum(ind)))

    weight = np.cos(np.deg2rad(vo['lat']))
    # replicate weight for each time snapshot
    weight = np.repeat(weight[np.newaxis, :], vo['data'].shape[0], axis=0)

    # use masked array to properly average the data
    data_mx = np.ma.masked_array(vo['data'], mask=ind)
    # use the same mask for weight
    weight_mx = np.ma.masked_array(weight, mask=ind)
    # average
    result = np.sum(data_mx * weight_mx, axis=1) / np.sum(weight_mx, axis=1)
    return result


print('SAGE 1020 OD Filled')
data_to_plot = get_diag(sparc_od_vo)

plt.plot(sparc_od_vo['time'], data_to_plot, '-o', ms=4,  label='Stratospheric Optical Depth (Filled)')

# integrate extinction vertically to get AOD
dz = 0.5  # km
asap_aod_1020_vo = {}
asap_aod_1020_vo['data'] = np.nansum(sparc_profile_filled_vo['data'][:, :, :, 1] * dz, axis=2)
# Take care of the unrealistic data
ind = asap_aod_1020_vo['data'] > 10
asap_aod_1020_vo['data'][ind] = np.NaN
asap_aod_1020_vo['lat'] = sparc_profile_filled_vo['lat']
print('{} unrealistic values (>10) of the column AOD where ignored in the SAGE Filled'.format(np.sum(ind)))

print('SAGE 1020 Filled profile')
data_to_plot = get_diag(asap_aod_1020_vo)
plt.plot(sparc_profile_filled_vo['time'], data_to_plot, label='Altitude/Latitude, Interpolation, Filled')

# v2 with exponent

# integrate extinction vertically to get AOD
dz = 0.5  # km
asap_aod_1020_vo = {}
asap_aod_1020_vo['data'] = np.nansum(10 ** sparc_profile_filled_vo['data'][:, :, :, 1] * dz, axis=2)
# Take care of the unrealistic data
ind = asap_aod_1020_vo['data'] > 10
asap_aod_1020_vo['data'][ind] = np.NaN
asap_aod_1020_vo['lat'] = sparc_profile_filled_vo['lat']
print('{} unrealistic values (>10) of the column AOD where ignored in the SAGE Filled'.format(np.sum(ind)))

print('SAGE 1020 Filled profile')
data_to_plot = get_diag(asap_aod_1020_vo)
plt.plot(sparc_profile_filled_vo['time'], data_to_plot, label='Altitude/Latitude, Interpolation, Filled')

plt.xlabel('Time')
plt.ylabel('AOD')
plt.title('ASAP AOD at 1.02 um')
plt.legend()
save_figure_bundle(pics_output_folder, 'asap AOD ts')


# Plot the Hovmoeller diagrams of the column AOD
# Note bright Yellow spots, which highlight the erroneous data and which was masked in the time series averaging
fig, axes = plt.subplots(constrained_layout=True, nrows=2, ncols=1, figsize=(15, 10))
contour_levels = np.linspace(0, 0.2, 21)
plt.sca(axes[0])
plt.contourf(sparc_od_vo['time'], sparc_od_vo['lat'], sparc_od_vo['data'].transpose(),
             contour_levels, extend='both')
plt.colorbar()
plt.xlim((dt.datetime(1979, 1, 1), dt.datetime(2005, 1, 1)))
plt.title('Stratospheric Optical Depth (Filled)')

plt.sca(axes[1])
plt.contourf(sparc_profile_filled_vo['time'], sparc_profile_filled_vo['lat'],
             asap_aod_1020_vo.transpose(),
             contour_levels, extend='both')
plt.colorbar()
plt.xlabel('Time')
plt.ylabel('Latitude')
plt.title('Altitude/Latitude, Interpolation, Filled')
plt.xlim((dt.datetime(1979, 1, 1), dt.datetime(2005, 1, 1)))

plt.suptitle('SPARC ASAP AOD at 1.02 um from different data sets', fontsize=24)
save_figure_bundle(pics_output_folder, 'asap AOD howmoller')

diff = sparc_od_vo['data'][69:] - asap_aod_1020_vo[:-24]
plt.figure()
plt.contourf(diff, np.linspace(0, 0.01, 11), extend='both')
plt.colorbar()


# time evolution, snapshots

def plot_asap_all_snapshots(var_name, var_column_index, contour_levels):
    plt.figure(figsize=(10, 5))

    for time_index in range(sparc_profile_filled_vo['time'].shape[0]):
        plt.clf()

        # data shape is data = np.zeros((len(files_list), 32, 70, 28))
        time_data = sparc_profile_filled_vo['time']
        lat_data = sparc_profile_filled_vo['lat']
        alt_data = sparc_profile_filled_vo['altitude']

        data_to_plot = sparc_profile_filled_vo['data'][time_index, :, :, var_column_index]
        plt.contourf(lat_data, alt_data, data_to_plot.transpose(), contour_levels, cmap='hot_r',
                     extend='both')  # spectral_r
        plt.colorbar()
        plt.xlabel('Latitude')
        plt.ylabel('Altitude, (km)')
        plt.title(var_name + ', ' + time_data[time_index].strftime("%b, %Y"))
        plt.tight_layout()

        save_figure(pics_output_folder + '/asap/' + var_name + '/', 'asap_' + '%02d' % (time_index) + '.png')


# column 1 is ext at 1020 nm
contour_levels = np.linspace(0, 4, 21) * 10 ** -3
plot_asap_all_snapshots('1020 nm extinction', 1, contour_levels)

contour_levels = np.linspace(0, 20, 21)
plot_asap_all_snapshots('Water vapor mixing ratio (ppm)', 27, contour_levels)

contour_levels = np.linspace(190, 270, 21)
plot_asap_all_snapshots('Temperature (K)', 25, contour_levels)

print('done')
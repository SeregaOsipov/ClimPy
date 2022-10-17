import xarray as xr
import datetime as dt
import cartopy.crs as crs
import numpy as np
from climpy.utils.file_path_utils import get_root_storage_path_on_hpc
from matplotlib.colors import LinearSegmentedColormap
from climpy.utils.plotting_utils import save_figure
# import matplotlib.pyplot as plt
# from climpy.utils.merra_utils import derive_merra2_pressure_stag_profile

data_root = '/work/mm0062/b302074/Data/NASA/MERRA2/'

#%% derive fractions for each aerosol type


def get_dust_fractions(date_to_process):
    natural_fraction = 1
    anthropogenic_fraction = 0
    return natural_fraction, anthropogenic_fraction


def get_sea_salt_fractions(date_to_process):
    natural_fraction = 1
    anthropogenic_fraction = 0
    return natural_fraction, anthropogenic_fraction


def get_bc_fractions(date_to_process):
    natural_fraction = 1
    anthropogenic_fraction = 0
    return natural_fraction, anthropogenic_fraction


def get_oc_fractions(date_to_process):
    natural_fraction = 1
    anthropogenic_fraction = 0
    return natural_fraction, anthropogenic_fraction


def get_sulfate_fractions(date_to_process):
    '''
    The script assumes that so4 above tropopause is natural and below is anthropogenic
    Tropopause is set fixed, but can be improved by using online diagnosed tropopause height from MERRA2 itself
    Use TROPPB in tavg1_2d_slv_Nx for that
    :return:
    '''
    fp = '{}/{}/MERRA2_100.{}.{}.nc4'.format(data_root, 'inst3_3d_aer_Nv', 'inst3_3d_aer_Nv', date_to_process.strftime('%Y%m%d'))
    xr_3d = xr.open_dataset(fp)

    fp = '{}/{}/MERRA2_100.{}.{}.nc4'.format(data_root, 'tavg3_3d_nav_Ne', 'tavg3_3d_nav_Ne', date_to_process.strftime('%Y%m%d'))
    xr_3d_nav = xr.open_dataset(fp)  # provides stag grid

    dz = -1*xr_3d_nav['ZLE'].diff(dim='lev', )

    # have to derive column loading first
    so4 = xr_3d['SO4'] * xr_3d['AIRDENS'] * dz.to_numpy()  # mmr -> loading in kg/m^2 (= concentration * dz)
    # so4 = so4.mean(dim=('time'))  # get daily average
    # split so4 above and below tropopause
    # p_stag = derive_merra2_pressure_stag_profile(xr_3d)
    # p_stag[:, 14].mean() is ~= 113

    tropopause_index = 40  # ~ 13.6 km
    so4_top = so4.where(so4.lev <= tropopause_index)  # remember that z profile is inverted in MERRA2 (level 0 is TOP)
    so4_bottom = so4.where(so4.lev > tropopause_index)

    natural_fraction = so4_top.sum(dim=('lev'))/so4.sum(dim=('lev'))
    anthropogenic_fraction = so4_bottom.sum(dim=('lev'))/so4.sum(dim=('lev'))

    return natural_fraction.rename('sulfate_natural_fraction'), anthropogenic_fraction.rename('sulfate_anth_fraction')


#%%


def derive_aod_fractions(date_to_process, keys, anth_fracs, natural_fracs):
    '''
    apply speciated fractions to the AOD and derive natural/anthropogenic fractions
    keys: nc AOD vars
    anth_fracs & natural_fracs: corresponding speciated fractions
    :return:
    '''

    fp = '{}/{}/MERRA2_100.{}.{}.nc4'.format(data_root, 'tavg1_2d_aer_Nx', 'tavg1_2d_aer_Nx', date_to_process.strftime('%Y%m%d'))
    ds = xr.open_dataset(fp)

    # check the summation
    aod_sum = ds['DUEXTTAU'] + ds['SSEXTTAU'] + ds['SUEXTTAU'] + ds['OCEXTTAU'] +  + ds['BCEXTTAU']
    diff = (aod_sum - ds['TOTEXTTAU']) / ds['TOTEXTTAU']
    print('Largest relative difference is component-wise AOD sum is {} %'.format(100*diff.max().item(0)))

    anth_aod = 0
    natural_aod = 0
    for key, anth_frac, natural_frac in zip(keys, anth_fracs, natural_fracs):
        # here we have 3-hourly and 1-hourly: have to downsample or upsample
        aod_3hourly = ds[key].resample(time='3H').mean()
        anth_aod += aod_3hourly * anth_frac
        natural_aod += aod_3hourly * natural_frac

    anth_aod_frac = anth_aod / (anth_aod+natural_aod)
    natural_aod_frac = 1 - anth_aod_frac

    return anth_aod_frac.rename('anth_aod_fraction'), natural_aod_frac.rename('natural_aod_fraction'), anth_aod.rename('anth_aod'), natural_aod.rename('natural_aod')


#%%
if __name__ == '__main__':
    pics_output_folder_root = get_root_storage_path_on_hpc() + '/Pictures/Papers/Zittis/'
    date_to_process = dt.datetime(1991, 6, 15)

    sulfate_natural_frac, sulfate_anth_frac = get_sulfate_fractions(date_to_process)
    sea_salt_natural_frac, sea_salt_anth_frac = get_sea_salt_fractions(date_to_process)
    dust_natural_frac, dust_anth_frac = get_dust_fractions(date_to_process)
    oc_natural_frac, oc_anth_frac = get_oc_fractions(date_to_process)
    bc_natural_frac, bc_anth_frac = get_bc_fractions(date_to_process)

    keys = ('DUEXTTAU', 'SSEXTTAU', 'SUEXTTAU', 'OCEXTTAU', 'BCEXTTAU')
    anth_fracs = (dust_anth_frac, sea_salt_anth_frac, sulfate_anth_frac, oc_anth_frac, bc_anth_frac)
    natural_fracs = (dust_natural_frac, sea_salt_natural_frac, sulfate_natural_frac, oc_natural_frac, bc_natural_frac)

    anth_aod_frac, natural_aod_frac, anth_aod, natural_aod = derive_aod_fractions(date_to_process, keys, anth_fracs, natural_fracs)

#%% Plot diags
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('TkAgg')

    palette = ['black', 'blue', 'purple', 'cyan', 'green', 'yellow', 'red']
    palette = ['white', 'blue', 'purple', 'cyan', 'green', 'yellow', 'red']  # drop black as it makes difficult to see coastlines
    cm = LinearSegmentedColormap.from_list('GEE', palette, N=256)

    data_to_plot = anth_aod_frac.mean(dim='time')
    fig = plt.figure(figsize=(3840 / 100 / 2, 2160 / 100 / 2), constrained_layout=True)
    ax = plt.axes(projection=crs.PlateCarree())
    ax.coastlines()
    cl = np.linspace(np.nanpercentile(data_to_plot, 5), np.nanpercentile(data_to_plot, 97), 10)
    plt.contourf(ds.lon, ds.lat, data_to_plot, levels=cl, extend='both', cmap=cm)
    plt.colorbar()
    plt.title('AOD anthropogenic fraction')
    save_figure(pics_output_folder_root, 'aod anth frac {}'.format(date_to_process.strftime('%Y%m%d')))

    data_to_plot = natural_aod_frac.mean(dim='time')
    fig = plt.figure(figsize=(3840 / 100 / 2, 2160 / 100 / 2), constrained_layout=True)
    ax = plt.axes(projection=crs.PlateCarree())
    ax.coastlines()
    cl = np.linspace(np.nanpercentile(data_to_plot, 5), np.nanpercentile(data_to_plot, 97), 10)
    plt.contourf(ds.lon, ds.lat, data_to_plot, levels=cl, extend='both', cmap=cm)
    plt.colorbar()
    plt.title('AOD natural fraction')
    save_figure(pics_output_folder_root, 'aod natural frac {}'.format(date_to_process.strftime('%Y%m%d')))
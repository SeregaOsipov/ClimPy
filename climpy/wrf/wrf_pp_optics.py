import netCDF4
import numpy as np
import os
import pandas as pd
import wrf as wrf
from climpy.utils.atmos_utils import compute_column_from_vmr_profile
import xarray as xr
import argparse
from climpy.utils.wrf_chem_utils import get_aerosols_keys
from climpy.utils.wrf_chem_made_utils import get_wrf_size_distribution_by_modes
from climpy.utils.refractive_index_utils import mix_refractive_index
import climpy.utils.mie_utils as mie

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

'''
This script derives optical properties for WRF output 
Currently only column AOD for MADE only (log-normal pdfs)
'''

parser = argparse.ArgumentParser()
parser.add_argument("--mode", "--port", "--host", help="pycharm")
parser.add_argument("--wrf_in", help="wrf input file path", default='/Users/osipovs/Data/AirQuality/EMME/2050/EDGAR_trend/chem_100_v1/output/pp_optics/merge/wrfout_d01_timmean') #required=True)# default='/work/mm0062/b302074/Data/AirQuality/AQABA/chem_100_v23/output/pp_wrf_optics/merge/wrfout_d01_monmean_regrid')
parser.add_argument("--wrf_out", help="wrf output file path", default='/Users/osipovs/Data/AirQuality/EMME/2050/EDGAR_trend/chem_100_v1/output/pp_optics/merge/wrfout_d01_timmean_optics') #required=True)#default='/work/mm0062/b302074/Data/AirQuality/AQABA/chem_100_v23/output/pp_wrf_optics/merge/wrfout_d01_monmean_regrid_optics')
args = parser.parse_args()

print('Will process this WRF:\nin {}\nout {}'.format(args.wrf_in, args.wrf_out))

chem_opt = 100

# setup wl grid for optical props
default_wrf_wavelengths = np.array([0.3, 0.4, 0.5, 0.6, 0.999])  # default WRF wavelengths in SW
maritime_wavelengths = np.array([380, 440, 500, 675, 870])/10**3  # aeronet
wavelengths = np.unique(np.append(default_wrf_wavelengths, maritime_wavelengths))  # Comprehensive wl grid

wavelengths = np.array([0.55, ])  # short & quick
#%%


def export_to_netcdf(var, mode='a', is_var_time_dependent=True):
    # have to remove attributes so that xarray can save data to netcdf
    if 'projection' in var.attrs:
        del var.attrs['projection']
    if 'coordinates' in var.attrs:
        del var.attrs['coordinates']

    mode = 'a'  # set mode to append, time needs to be record dimensions (forced by unlimited) to allow ncrcat
    if not os.path.exists(args.wrf_out):
        mode = 'w'

    unlimited_dim = None
    if is_var_time_dependent:
        unlimited_dim = xr_in['so4aj'].dims[0]  # time

    # load should fix the slow writing
    var.load().to_netcdf(args.wrf_out, mode=mode, unlimited_dims=unlimited_dim, format='NETCDF4_CLASSIC')


# nc_in = netCDF4.Dataset(args.wrf_in)
xr_in = xr.open_dataset(args.wrf_in)
xr_in = xr_in.rename({'XTIME': 'time', 'XLAT':'lat', 'XLONG':'lon'})#, 'west_east':'lon', 'south_north':'lat'})
#%% Derive spectral column AOD
ri_ds = mix_refractive_index(xr_in, chem_opt, wavelengths)  # volume weighted RI
# ri_ds = ri_ds.rename({'XTIME': 'time', 'XLAT':'lat', 'XLONG':'lon', 'west_east':'lon', 'south_north':'lat'})
dA_ds = get_wrf_size_distribution_by_modes(xr_in, moment='dA', sum_up_modes=True, column=True)
# dA_ds = dA_ds.rename({'XTIME': 'time', 'XLAT':'lat', 'XLONG':'lon'})
#%% serial implementation


def derive_aerosols_optical_properties(ri_ds, dA_ds):
    '''
    Use this for a single aerosols type and loop through the list
    Currently only extinction / optical depth

    :param ri_ds: RI of the aerosols
    :param dA_ds: cross-section area distribution
    :return:
    '''

    # Compute Mie extinction coefficients
    # dims are time, r, wl
    qext = np.zeros(dA_ds['dAdlogd'].shape + ri_ds['wavelength'].shape)
    qext[:] = np.NaN
    time_list = []
    for time_index in range(dA_ds.time.size):
        lat_list = []
        for lat_index in range(dA_ds.south_north.size):  # range(5): #
            print(lat_index)
            lon_list = []
            for lon_index in range(dA_ds.west_east.size):
                mie_ds = mie.get_mie_efficiencies(ri_ds.isel(time=time_index, south_north=lat_index, west_east=lon_index).ri, dA_ds['radius'], ri_ds['wavelength'])
                qext[time_index, lat_index, lon_index] = np.swapaxes(mie_ds['qext'], 0, 1)
                lon_list += [mie_ds, ]
                # ds = mie_ds.expand_dims(dim=['time', 'south_north', 'west_east'])
                # ds = ds.assign_coords(lon=(('south_north', 'west_east'), dA_ds.lon.isel(west_east=lon_index, south_north=lat_index).data.reshape(1,1)))
                # ds = ds.assign_coords(lat=(['south_north', 'west_east'], dA_ds.lat.isel(west_east=lon_index, south_north=lat_index).data.reshape(1,1)))
                # ds = ds.assign_coords(time=('time', dA_ds.time.isel(time=time_index).data.reshape(1)))
                # ds = ds.rename({'lon': 'west_east', 'lat':'south_north'})
            lat_list += [xr.concat(lon_list, dim='west_east'), ]
        time_list+=[xr.concat(lat_list, dim='south_north'), ]
    qext_ds = xr.concat(time_list, dim='time')

    # dims: time, r, wl & time, r
    # integrand = qext * dA_ds['dAdlogd'][..., np.newaxis]
    integrand = qext_ds * dA_ds['dAdlogd']
    integrand = integrand.assign_coords(log_radius=('radius', np.log(dA_ds['radius'].data)))
    # column_od_ds = np.trapz(integrand, np.log(dA_ds['radii']), axis=-2)  # sd is already dAdlnr
    column_od_ds = integrand.qext.integrate(coord='log_radius')
    # column_od = np.sum(column_od_by_modes, axis=1)  # sum up modes

    return column_od_ds


# debugging subset
# dsize=15
# ri_ds = ri_ds.isel(south_north=slice(0,dsize), west_east=slice(0,dsize))
# dA_ds = dA_ds.isel(south_north=slice(0,dsize), west_east=slice(0,dsize))
column_od_ds = derive_aerosols_optical_properties(ri_ds, dA_ds)

column_od_ds.to_netcdf(args.wrf_out)#, mode=mode, unlimited_dims=unlimited_dim, format='NETCDF4_CLASSIC')

print('DONE')
exit()

# code below is not finished

#%% Export

var_key = 'AOD'
dims = (xr_in['ALT'].dims[0],) + xr_in['ALT'].dims[2:] + ('wavelength', )  # Unlimited dimension has to be first
data = xr.Dataset({var_key: (dims, column_od_ds)})

data[var_key].attrs["long_name"] = 'Spectral column AOD'
data[var_key].attrs["units"] = ""
data[var_key].attrs["description"] = "Derived using Mie and volume weighted RIs"
export_to_netcdf(data)

var_key = 'wavelengths'  # export wavelength grid
dims = ('wavelength', )
data = xr.Dataset({var_key: (dims, wavelengths)})
data[var_key].attrs["long_name"] = 'Wavelengths'
data[var_key].attrs["units"] = "um"
data[var_key].attrs["description"] = "Rho grid (centers of the interval)"
export_to_netcdf(data, is_var_time_dependent=False)

print('PMs are done, proceed to coordinate variables')
# export_to_netcdf(wrf_ds[['XTIME', 'XLONG', 'XLAT']])  # export time/lat/lon as well
export_to_netcdf(xr_in[['XTIME', 'lon', 'lat']])  # export time/lat/lon as well

print('DONE')


#%% ray implementation

#
# def thread_safe_calculations(ri, r_data, wavelength):  # function for ray
#     mie_vo = mie.get_mie_efficiencies(ri, r_data, wavelength)
#     return np.swapaxes(mie_vo['qext'], 0, 1)  # qext[time_index, lat_index, lon_index]
#
#
# def derive_aerosols_optical_properties(ri_vo, dA_vo):
#     '''
#     Use this for a single aerosols type and loop through the list
#     Currently only extinction / optical depth
#
#     :param ri_vo: RI of the aerosols
#     :param dA_vo: cross-section area distribution
#     :return:
#     '''
#
#     # Compute Mie extinction coefficients
#     # dims are time, r, wl
#     qext = np.zeros(dA_vo['data'].shape + ri_vo['wl'].shape)
#
#     for time_index in range(qext.shape[0]):
#         print('time index: {}'.format(time_index))
#         for lat_index in qext.shape[1]:  # range(5): #
#             print(lat_index)
#             output_ids = []
#             for lon_index in range(qext.shape[2]):
#                 ri, r_data, wavelength = ri_vo['ri'][time_index, lat_index, lon_index], dA_vo['radii'], ri_vo['wl']
#                 # mie_vo = mie.get_mie_efficiencies(ri_vo['ri'][time_index, lat_index, lon_index], dA_vo['radii'], ri_vo['wl'])
#                 # qext[time_index, lat_index, lon_index] = np.swapaxes(mie_vo['qext'], 0, 1)
#                 output_ids.append(thread_safe_calculations.remote(ri, r_data, wavelength))
#
#             output_list = ray.get(output_ids)
#             qext[time_index, lat_index, :] = np.array(output_list)
#
#     # dims: time, r, wl & time, r
#     integrand = qext * dA_vo['data'][..., np.newaxis]
#     column_od = np.trapz(integrand, np.log(dA_vo['radii']), axis=-2)  # sd is already dAdlnr
#     # column_od = np.sum(column_od_by_modes, axis=1)  # sum up modes
#
#     return column_od
#
#
# column_od = derive_aerosols_optical_properties(ri_vo, dA_vo)

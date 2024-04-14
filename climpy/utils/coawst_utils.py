import time

import netCDF4
import numpy as np
from matplotlib import pyplot as plt

from climpy.utils.wrf_utils import derive_wrf_net_flux_from_accumulated_value, derive_wrf_net_flux_from_instanteneous_value, generate_netcdf_uniform_time_data
from libs.ROMS.RomsController import get_roms_nc_parameters
from libs.readers.AbstractNetCdfReader import process_grid_area_range_impl
from climpy.utils.time_utils import process_time_range_impl

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'



def read_roms_variables(nc_file_path, nc_pp_file_path, nc_grid_file_path, time_range_vo, list_of_3d_vars, list_of_2d_vars, list_of_2d_vars_pp=None, list_of_grid_vars=()):
    # remember that depth in ROMS is negative
    Vtransform, Vstretching, sc_r, Cs_r, sc_w, Cs_w = get_roms_nc_parameters(nc_file_path)

    netcdf_dataset_impl = netCDF4.Dataset
    if (type(nc_file_path) is list):
        netcdf_dataset_impl = netCDF4.MFDataset

    nc = netcdf_dataset_impl(nc_file_path)
    if ( len(list_of_2d_vars_pp) > 0 ):
        nc_pp = netcdf_dataset_impl(nc_pp_file_path)
    if ( len(list_of_grid_vars) > 0 ):
        nc_grid = netCDF4.Dataset(nc_grid_file_path)
    time_variable = nc.variables['ocean_time']
    rawTimeData = generate_netcdf_uniform_time_data(time_variable)
    t_slice, time_data = process_time_range_impl(rawTimeData[:], time_range_vo)

    # construct the slices
    n_levels = len(sc_r)
    z_slice = slice(n_levels - 1, n_levels)
    x_slice, y_slice, yxSlicesArray, tyxSlicesArray, tzyxSlicesArray = get_roms_red_sea_slices(nc.variables['mask_rho'][:], z_slice, t_slice)

    mask_rho = nc.variables['mask_rho'][yxSlicesArray]

    plt.figure()
    plt.contour(mask_rho)
    plt.title('slices range')


    h = nc.variables['h'][yxSlicesArray]

    # z = compute_depths(nc_file_path, tyxSlicesArray)
    lon_rho = nc.variables['lon_rho'][yxSlicesArray]
    lat_rho = nc.variables['lat_rho'][yxSlicesArray]

    roms_output_set = {}
    # roms_output_set['z'] = z

    # roms_output_set['temp'] = nc.variables['temp'][t_slice, :, y_slice, x_slice].squeeze()
    # print 'temprorary read the potential temperature'

    for var_index in range(len(list_of_3d_vars)):
        current_var_name = list_of_3d_vars[var_index]
        t1 = time.time()
        roms_output_set[current_var_name] = nc.variables[current_var_name][tzyxSlicesArray].squeeze()
        #there are several isolate point that needs to be remove
        roms_output_set[current_var_name].mask[:, 989 - 235, 59 - 4] = True
        roms_output_set[current_var_name].mask[:, 788 - 235, 158 - 4] = True
        roms_output_set[current_var_name].mask[:, 476 - 235, 153 - 4] = True
        # roms_output_set[current_var_name].mask[:, 202, 182]

        # roms_output_set[current_var_name] = roms_output_set[current_var_name]
        t2 = time.time()

        if ( current_var_name == 'rho'):
            roms_output_set[current_var_name] += nc.variables['rho0']

        print('reading ' + current_var_name + ' ' + str(t2 - t1))

    for var_index in range(len(list_of_2d_vars)):
        current_var_name = list_of_2d_vars[var_index]
        t1 = time.time()
        roms_output_set[current_var_name] = nc.variables[current_var_name][tyxSlicesArray].squeeze()
        # there are several isolate point that needs to be remove
        roms_output_set[current_var_name].mask[:, 989 - 235, 59 - 4] = True
        roms_output_set[current_var_name].mask[:, 788 - 235, 158 - 4] = True
        roms_output_set[current_var_name].mask[:, 476 - 235, 153 - 4] = True
        # roms_output_set[current_var_name].mask[:, :, 202, 182]
        # roms_output_set[current_var_name] = roms_output_set[current_var_name].squeeze()
        t2 = time.time()

        print('reading nc: ' + current_var_name + ' ' + str(t2 - t1))

    for var_index in range(len(list_of_2d_vars_pp)):
        current_var_name = list_of_2d_vars_pp[var_index]
        if (current_var_name == 'evap_minus_precip'):
            roms_output_set[current_var_name] = nc.variables['evaporation'][tyxSlicesArray].squeeze() - nc.variables['rain'][tyxSlicesArray].squeeze()
            continue
        t1 = time.time()
        roms_output_set[current_var_name] = np.ma.array(nc_pp.variables[current_var_name][tyxSlicesArray]).squeeze()
        # there are several isolate point that needs to be remove
        roms_output_set[current_var_name][:, 989 - 235, 59 - 4] = np.NaN
        roms_output_set[current_var_name][:, 788 - 235, 158 - 4] = np.NaN
        roms_output_set[current_var_name][:, 476 - 235, 153 - 4] = np.NaN
        # roms_output_set[current_var_name][:, :, 202, 182]
        # roms_output_set[current_var_name] = roms_output_set[current_var_name].squeeze()
        t2 = time.time()

        print('reading nc pp: ' + current_var_name + ' ' + str(t2 - t1))

    for var_index in range(len(list_of_grid_vars)):
        current_var_name = list_of_grid_vars[var_index]
        t1 = time.time()
        roms_output_set[current_var_name] = nc_grid.variables[current_var_name][yxSlicesArray]
        t2 = time.time()
        print('reading grid nc: ' + current_var_name + ' ' + str(t2 - t1))

    current_var_name = 'mask_rho'
    # there are several isolate point that needs to be remove
    roms_output_set[current_var_name][989 - 235, 59 - 4] = 0
    roms_output_set[current_var_name][788 - 235, 158 - 4] = 0
    roms_output_set[current_var_name][476 - 235, 153 - 4] = 0
    # roms_output_set[current_var_name].mask[:, :, 202, 182]

    # t1 = time.time()
    # # pres1 = gbs.p_from_z(z[0,:,:], lat_rho)
    # pressure = csr.pres(-1*z, lat_rho)
    # temp = csr.temp(salinity, potTemp, pressure).squeeze()
    # # specificHeatCapacity = csr.cp(salinity, temp, pressure).squeeze()
    # t2 = time.time()
    # print 'seawater ' + str(t2-t1)

    # print "applying manual spatial mask based on salinity threshold"
    # # manually mask some of the points, they are anomalous
    # print np.max(roms_output_set['salt'])
    # ind = roms_output_set['salt'] > 60
    #
    # potTemp.mask = np.logical_or(potTemp.mask, ind)
    # salinity.mask = np.logical_or(salinity.mask, ind)
    # rho.mask = np.logical_or(rho.mask, ind)
    #
    # swrad.mask = np.logical_or(swrad.mask, ind)
    # lwrad.mask = np.logical_or(lwrad.mask, ind)
    # # sensible.mask = np.logical_or(sensible.mask, ind)
    # latent.mask = np.logical_or(latent.mask, ind)
    # svstr.mask = np.logical_or(svstr.mask, ind)
    # zeta.mask = np.logical_or(zeta.mask, ind)
    #
    # evap_minus_precip.mask = np.logical_or(evap_minus_precip.mask, ind)
    # shflux.mask = np.logical_or(shflux.mask, ind)
    #
    # mld.mask = np.logical_or(mld.mask, ind)
    # ohc.mask = np.logical_or(ohc.mask, ind[:, np.newaxis])
    # print np.max(salinity)
    # print 'mask has been applied'


    roms_output_set['time_data'] = time_data
    roms_output_set['lon_rho'] = lon_rho
    roms_output_set['lat_rho'] = lat_rho
    roms_output_set['mask_rho'] = mask_rho
    roms_output_set['h'] = h

    return roms_output_set
def read_wrf_rad_fluxes(file_path, time_range_vo, area_range_vo=None, use_accumulated_fluxes=True, double_call_diags=False):
    output_set = {}

    nc = netCDF4.MFDataset(file_path)
    nc_first = netCDF4.Dataset(file_path[0])
    # time_variable = netCDF4.MFTime(nc.variables['Times'])
    time_variable = nc.variables['Times']
    rawTimeData = generate_netcdf_uniform_time_data(time_variable)
    time_delta = rawTimeData[1]-rawTimeData[0]

    # daily_mean_number_of_steps = 24.0 / ( (time_delta.days*24+time_delta.seconds/60/60))
    timeInd = np.logical_and(rawTimeData >= time_range_vo.startDate, rawTimeData < time_range_vo.endDate)
    timeInd = np.where(timeInd)[0]
    #change the size of the data so that we can do daily averages
    # timeInd = timeInd[0:timeInd.shape[0]/8*8]
    time_data = rawTimeData[timeInd]
    # time_data = time_data.reshape((-1, daily_mean_number_of_steps)) #8
    # time_data = time_data[:, daily_mean_number_of_steps/2]
    output_set['time_data'] = time_data

    # construct the slices
    tSlice = slice(timeInd[0], timeInd[-1] + 1)
    # 0 for land
    # y_slice = slice(35, 196)
    # x_slice = slice(26, 120)

    if ( area_range_vo is None):
        y_slice = slice(0, nc.dimensions['south_north'].size + 1)
        x_slice = slice(0, nc.dimensions['west_east'].size + 1)
    else:
        x_slice, y_slice, dummy1, dummy2 = process_grid_area_range_impl(nc.variables['XLAT'][0], nc.variables['XLONG'][0], area_range_vo)

    z_slice = slice(0, nc.dimensions['bottom_top'].size+1)# all layers

    tyxSlicesArray = [tSlice, y_slice, x_slice]
    tzyxSlicesArray = [tSlice, z_slice, y_slice, x_slice]
    t0yxSlicesArray = [slice(1, 2), y_slice, x_slice]

    print("starting to read variables")

    output_set['lat'] = nc.variables['XLAT'][t0yxSlicesArray]
    output_set['lon'] = nc.variables['XLONG'][t0yxSlicesArray]

    land_mask = nc.variables['LANDMASK'][t0yxSlicesArray]
    land_mask = land_mask.squeeze()
    output_set['LANDMASK'] = land_mask
    # sst = nc.variables['SST'][tyxSlicesArray]
    # sst = compute_wrf_daily_data(sst)
    # output_set['SST'] = np.ma.array(sst)
    # output_set['SST'].mask = land_mask[np.newaxis, :, :]

    #also derive wind speed at 10m to estimate the impact on latent heat flux
    #we take the magnitude to avoid averaging of the daily cycle due to direction

    # u10 = compute_wrf_daily_data(np.absolute(nc.variables['U10'][tyxSlicesArray]))
    # v10 = compute_wrf_daily_data(np.absolute(nc.variables['V10'][tyxSlicesArray]))
    # wind10 =np.sqrt(u10**2+v10**2)
    # output_set['wind10'] = np.ma.array(wind10)
    # output_set['wind10'].mask = land_mask[np.newaxis, :, :]
    # output_set['|u10|'] = np.ma.array(u10)
    # output_set['|u10|'].mask = land_mask[np.newaxis, :, :]
    # output_set['|v10|'] = np.ma.array(v10)
    # output_set['|v10|'].mask = land_mask[np.newaxis, :, :]

    #and specific humidity at 2m
    # q2 = compute_wrf_daily_data(nc.variables['Q2'][tyxSlicesArray])
    # output_set['specific_humidity_2m'] = np.ma.array(q2/(1+q2))
    # output_set['specific_humidity_2m'].mask = land_mask[np.newaxis, :, :]

    # t2 = compute_wrf_daily_data(nc.variables['T2'][tyxSlicesArray])
    # output_set['T2'] = np.ma.array(t2)
    # output_set['T2'].mask = land_mask[np.newaxis, :, :]


    # accumulated fluxes sometimes are buggy in the WRF, keep it in mind

    derive_net_flux_impl = derive_wrf_net_flux_from_instanteneous_value
    A_I_flag = '' #accumulate or instanteneous fluxes
    if ( use_accumulated_fluxes ):
        derive_net_flux_impl = derive_wrf_net_flux_from_accumulated_value
        A_I_flag = 'AC'
        #because fluxes from the accumulated values are computed as a difference, drop the first time element to keep consistent sizes
        output_set['time_data'] = output_set['time_data'][1:]

    bands_list = ('SW', 'LW')
    location_list = ('B', 'T') # bottom and top
    sky_list = ('', 'C') #all sky and clear sky

    perturbations_list = ('',)
    if (double_call_diags):
        # prepare the fluxes with and without SO2, sulf, dust
        perturbations_list += ('_NO_SO2', '_NO_SULF')


    for band in bands_list:
        for location in location_list:
            for sky in sky_list:
                for perturbation in perturbations_list: #loop for NO_SO2, NO_DUST and so on
                    flux_down_name = A_I_flag + band + 'DN' + location + sky + perturbation
                    flux_up_name = A_I_flag + band + 'UP' + location + sky + perturbation
                    net_flux, down_flux, up_flux = derive_net_flux_impl(nc, flux_down_name, flux_up_name, tyxSlicesArray, time_delta)

                    output_net_var_name = band+'NET'+location+sky+perturbation
                    output_set[output_net_var_name.lower()] = net_flux

                    output_down_var_name = band + 'DN' + location + sky+perturbation
                    output_set[output_down_var_name.lower()] = down_flux

                    output_up_var_name = band + 'UP' + location + sky+perturbation
                    output_set[output_up_var_name.lower()] = up_flux

    # output_set['sw_net_boa_clear_sky'] = sw_net_boa_clear_sky

    return output_set

def get_roms_red_sea_slices(mask_rho, z_slice, t_slice):
    #this one is sliced up to suez
    y_slice = slice(235, mask_rho.shape[0] - 155)
    #this one is full extend to the north
    y_slice = slice(235, mask_rho.shape[0]-1)
    x_slice = slice(4, 190)

    yxSlicesArray = [y_slice, x_slice]
    tyxSlicesArray = [t_slice, y_slice, x_slice]
    tzyxSlicesArray = [t_slice, z_slice, y_slice, x_slice]

    return x_slice, y_slice, yxSlicesArray, tyxSlicesArray, tzyxSlicesArray


def get_roms_red_sea_xsection_slices(mask_rho, z_slice, t_slice):
    #this one is full extend to the north
    # y_slice = slice(235, mask_rho.shape[0]-1)

    # this one is sliced up to suez
    y_slice = slice(235, mask_rho.shape[0] - 155)
    x_slice = slice(4 + 105, 4 + 105 + 1)

    yxSlicesArray = [y_slice, x_slice]
    tyxSlicesArray = [t_slice, y_slice, x_slice]
    tzyxSlicesArray = [t_slice, z_slice, y_slice, x_slice]

    return x_slice, y_slice, yxSlicesArray, tyxSlicesArray, tzyxSlicesArray



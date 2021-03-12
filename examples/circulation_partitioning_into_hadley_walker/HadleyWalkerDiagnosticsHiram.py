import netCDF4
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
from Projects.HadleyWalker.HadleyWalkerSeparationController import HadleyWalkerSeparationController
from libs.picture_utils import setup_plot_styles
from climpy.plotting_utils import save_fig
from libs.ranges.AreaRangeVO import AreaRangeVO
from libs.readers.DataSetDescriptionVO import DataSetDescriptionVO
from libs.readers.DataSetReadersFactory import DataSetReadersFactory

plt.ion()
setup_plot_styles()
pics_output_folder = '/home/osipovs/Pictures/Projects/HadleyWalkerCirculation/'
separationController = HadleyWalkerSeparationController()

plt.ioff()

#this is the setup for the ensemble mean
# nc_omega_name_list = ('omega', 'omega', 'V_VEL_L100_Avg', 'V_VEL_L100_Avg')
# nc_z_name_list = ('hght', 'hght', 'HGT_L100_Avg', 'HGT_L100_Avg')
# nc_level_name_list = ('level', 'level', 'lev', 'lev')
#
# w_nc_fp_list = ('/shaheen/project/k1090/dogarm/k14/datasets/OMEGA_HIRAM/omega_hiram_ensmean_TotalPeriod_Climatology.nc',
#                 '/shaheen/project/k1090/dogarm/k14/datasets/OMEGA_HIRAM/omega_hiram_ensmean_ElChichonPinatubo_Climatology.nc',
#                 '/shaheen/project/k1090/dogarm/k14/datasets/OMEGA_CFSR/Omega_CFSR_JJA_TotalPeriod_Climatology.nc',
#                 '/shaheen/project/k1090/dogarm/k14/datasets/OMEGA_CFSR/Omega_CFSR_JJA_ElChichonPinatubo_Climatology.nc'
#                 )
# z_nc_fp_list = ('/shaheen/project/k1090/dogarm/k14/datasets/GH_HIRAM/hght_hiram_ensmean_TotalPeriod_Climatology.nc',
#                 '/shaheen/project/k1090/dogarm/k14/datasets/GH_HIRAM/hght_hiram_ensmean_ElChichonPinatubo_Climatology.nc',
#                 '/shaheen/project/k1090/dogarm/k14/datasets/GH_CFSR/HGT_CFSR_JJA_TotalPeriod_Climatology.nc',
#                 '/shaheen/project/k1090/dogarm/k14/datasets/GH_CFSR/HGT_CFSR_JJA_ElChichonPinatubo_Climatology.nc'
#                 )

#this is the setup for the individual summers
nc_omega_name_list = ('omega', 'V_VEL_L100_Avg')
nc_z_name_list = ('hght', 'HGT_L100_Avg')
nc_level_name_list = ('level', 'lev')

w_nc_fp_list = ('/shaheen/project/k1090/dogarm/k14/datasets/OMEGA_HIRAM/omg_hiram_ensmean_JJA.nc',
                '/shaheen/project/k1090/dogarm/k14/datasets/OMEGA_CFSR/Omega_CFSR_JJA.nc',
                )
z_nc_fp_list = ('/shaheen/project/k1090/dogarm/k14/datasets/GH_HIRAM/hght_hiram_ensmean_JJA.nc',
                '/shaheen/project/k1090/dogarm/k14/datasets/GH_CFSR/HGT_CFSR_JJA.nc',
                )

labels_list = ('hiram', 'cfsr')

def prepare_data(w_nc_fp, z_nc_fp, omega_var_name, z_var_name, level_var_name):
    descVO = DataSetDescriptionVO(DataSetDescriptionVO.GENERIC_NETCDF)
    # descVO.levelsName = ''
    descVO.variableFilePath = w_nc_fp
    descVO.variableName = omega_var_name
    descVO.latName = 'lat'
    descVO.lonName = 'lon'
    descVO.levelsName = level_var_name


    readersFactory = DataSetReadersFactory()
    reader = readersFactory.getReader(descVO)
    # areaRangeVO = AreaRangeVO('belt', -60, 60, 0, 360)
    # eraDataSetVO = reader.readDataSet(descVO, areaRangeVO)
    wDataSetVO = reader.readDataSet(descVO)
    ind = np.abs(wDataSetVO.latData) <= 60
    wDataSetVO.latData = wDataSetVO.latData[ind]
    #keep only first time index and values at some single level
    # eraDataSetVO.fieldData = eraDataSetVO.fieldData[0,21,ind,:]
    wDataSetVO.fieldData = wDataSetVO.fieldData[:,:,ind,:]
    # wDataSetVO.fieldData = np.rollaxis(wDataSetVO.fieldData, 1)

    # descVO.variableName = 'u'
    # uDataSetVO = reader.readDataSet(descVO)
    # ind = np.abs(uDataSetVO.latData) <= 60
    # uDataSetVO.latData = uDataSetVO.latData[ind]
    # uDataSetVO.fieldData = uDataSetVO.fieldData[0,:,ind,:]
    # uDataSetVO.fieldData = np.rollaxis(uDataSetVO.fieldData, 1)
    #
    # descVO.variableName = 'v'
    # vDataSetVO = reader.readDataSet(descVO)
    # ind = np.abs(vDataSetVO.latData) <= 60
    # vDataSetVO.latData = vDataSetVO.latData[ind]
    # vDataSetVO.fieldData = vDataSetVO.fieldData[0,:,ind,:]
    # vDataSetVO.fieldData = np.rollaxis(vDataSetVO.fieldData, 1)

    descVO.variableFilePath = z_nc_fp
    descVO.variableName = z_var_name
    zDataSetVO = reader.readDataSet(descVO)
    ind = np.abs(zDataSetVO.latData) <= 60
    zDataSetVO.latData = zDataSetVO.latData[ind]
    zDataSetVO.fieldData = zDataSetVO.fieldData[:,:,ind,:]
    # zDataSetVO.fieldData = np.rollaxis(zDataSetVO.fieldData, 1)

    #get z from geopotential
    # zDataSetVO.fieldData /= 9.8

    return wDataSetVO, zDataSetVO
def save_data_to_netcdf(nc_out_file_path, lon_data, lat_data, level_data, time_data, diags_set_vo):
    print "saving into netcdf " + nc_out_file_path

    ncOut = netCDF4.Dataset(nc_out_file_path, 'w', format='NETCDF4_CLASSIC')

    timeDim = ncOut.createDimension('time', None)
    latDim = ncOut.createDimension('lat', lat_data.shape[0])
    lonDim = ncOut.createDimension('lon', lon_data.shape[0])
    levelDim = ncOut.createDimension('level', level_data.shape[0])

    timeVar = ncOut.createVariable('time', 'f8', ('time',))
    latVar = ncOut.createVariable('lat', 'f4', ('lat',))
    lonVar = ncOut.createVariable('lon', 'f4', ('lon',))

    var_names_list = ('massFlux_lambda', 'massFlux_teta', 'omega_lambda', 'omega_teta', 'psi_lambda', 'psi_teta')
    var_units_list = ('kg m^-2 s^-1', 'kg m^-2 s^-1', 'Pa s^-1', 'Pa s^-1', '?', '?')

    for i in range(len(var_names_list)):
        current_var_name = var_names_list[i]
        current_var = ncOut.createVariable(current_var_name, 'f4', ('time', 'level', 'lat', 'lon'))
        current_var.units = var_units_list[i]
        current_var[0,:] = diags_set_vo[current_var_name]

    latVar[:] = lat_data
    lonVar[:] = lon_data

    timeVar.units = 'hours since 0001-01-01 00:00:00'
    timeVar[:] = netCDF4.date2num(time_data, units= timeVar.units)

    ncOut.close()
def plot_averaged_diags_over_Africa(omega_data_set_vo, z_data, diags_set_vo):
    #extract Hadley and Walker over Africa and average it
    maxFlux = 5*10**-3
    minFlux = -maxFlux

    massFlux_lambda = diags_set_vo['massFlux_lambda']
    massFlux_teta = diags_set_vo['massFlux_teta']
    omega_lambda = diags_set_vo['omega_lambda']
    omega_teta = diags_set_vo['omega_teta']
    psi_lambda = diags_set_vo['psi_lambda']
    psi_teta = diags_set_vo['psi_teta']

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(25,10))
    #extract range over Africa only
    #because era lon starts from 0 we have to do the logical or
    ind = np.logical_or(omega_data_set_vo.lonData >= 360 - 20, omega_data_set_vo.lonData <= 60)
    currentMassFlux = np.nanmean(massFlux_lambda[:, :, ind], axis=2)
    # currentOmega = np.nanmean(omega_lambda[:, :, ind], axis=2)
    currentPsiLambda = np.nanmean(psi_lambda[:, :, ind], axis=2)
    currentZ = np.nanmean(z_data[:, :, ind], axis=2)
    currentZ = np.nanmean(currentZ, axis=1)

    # ind = np.logical_or(vDataSetVO.lonData >= 360-20, vDataSetVO.lonData <= 60)
    # currentV = np.nanmean(vDataSetVO.fieldData[:, :, ind], axis=2)
    # ind = np.logical_or(zDataSetVO.lonData >= 360-20, zDataSetVO.lonData <= 60)

    ax = axes[0]
    plt.sca(ax)
    pcolorHandle = plt.pcolor(omega_data_set_vo.latData, currentZ, currentMassFlux, cmap='bwr', vmin=minFlux, vmax=maxFlux)
    plt.colorbar(pcolorHandle)
    #pl.quiver(vDataSetVO.latData, vDataSetVO.levelsData, currentV, currentOmega)
    #draw zero mass flux contour
    # plotHandle = pl.contour(eraDataSetVO.latData, currentZ, currentMassFlux, levels = [0], colors='k',vmin=minFlux, vmax=maxFlux)

    #draw psi contours
    contourHandle = plt.contour(omega_data_set_vo.latData, currentZ, currentPsiLambda, linewidths=3)
    plt.colorbar(contourHandle, label='psi contour lines')

    #pl.gca().invert_yaxis()
    yLoc = currentZ[::2]
    yTicks = omega_data_set_vo.levelsData[::2]
    plt.yticks(yLoc, yTicks)
    # plt.ylim(currentZ[-1], currentZ[10])
    plt.xlabel('latitude')
    plt.ylabel('pressure (hPa)')
    plt.title('$m_{\lambda}$')

    ax = axes[1]
    plt.sca(ax)
    #extact only 35S to 10 N equatorial belt
    ind = np.logical_and(omega_data_set_vo.latData >= -35, omega_data_set_vo.latData <= 10)
    currentData = np.nanmean(massFlux_teta[:, ind, :], axis=1)
    # ind = np.logical_and(zDataSetVO.latData >= -35, zDataSetVO.latData <= 10)
    currentZ = np.nanmean(z_data[:, :, ind], axis=2)
    currentZ = np.nanmean(currentZ, axis=1)
    currentPsiTeta = np.nanmean(psi_teta[:, ind, :], axis=1)

    pcolorHandle = plt.pcolor(omega_data_set_vo.lonData, currentZ, currentData, cmap='bwr', vmin=minFlux, vmax=maxFlux)
    plt.colorbar(pcolorHandle)
    #draw zero mass flux contour
    # plotHandle = pl.contour(eraDataSetVO.lonData, currentZ, currentData, levels = [0], colors='k',vmin=minFlux, vmax=maxFlux)

    #draw psi contours
    contourHandle = plt.contour(omega_data_set_vo.lonData, currentZ, currentPsiTeta, linewidths=3)
    plt.colorbar(contourHandle, label='psi contour lines')
    yLoc = currentZ[::2]
    yTicks = omega_data_set_vo.levelsData[::2]
    plt.yticks(yLoc, yTicks)
    # plt.ylim(currentZ[-1], currentZ[10])

    plt.xlabel('longitude')
    plt.ylabel('pressure (hPa)')
    plt.title('$m_{\\theta}$')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, wspace=0.1)
    plt.suptitle('Hadley (20E-60W) & Walker (35S-10N)', fontsize=24)
def plot_individual_layer_diags(omega_data, levels_data, lon_data, lat_data, diags_set_vo, layer_index):
    U = diags_set_vo['U'][layer_index]
    massFlux_lambda = diags_set_vo['massFlux_lambda'][layer_index]
    massFlux_teta = diags_set_vo['massFlux_teta'][layer_index]
    omega_lambda = diags_set_vo['omega_lambda'][layer_index]
    omega_teta = diags_set_vo['omega_teta'][layer_index]
    psi_lambda = diags_set_vo['psi_lambda'][layer_index]
    psi_teta = diags_set_vo['psi_teta'][layer_index]

    plt.figure(figsize=(20,10))
    ax = plt.subplot(2, 2, 1)
    map = Basemap(ax = ax, projection='mill', resolution='l', llcrnrlon=areaRange.lon_min, llcrnrlat=areaRange.lat_min, urcrnrlon=areaRange.lon_max, urcrnrlat=areaRange.lat_max)
    map.drawcoastlines()
    x_m,y_m  = np.meshgrid(lon_data, lat_data)
    px, py = map(x_m, y_m)
    plotHandle = map.pcolormesh(px, py, U, cmap='bwr')
    map.colorbar(plotHandle)
    plt.title('poisson solution')

    ax = plt.subplot(2, 2, 2)
    map = Basemap(ax = ax, projection='mill', resolution='l', llcrnrlon=areaRange.lon_min, llcrnrlat=areaRange.lat_min, urcrnrlon=areaRange.lon_max, urcrnrlat=areaRange.lat_max)
    map.drawcoastlines()
    plotHandle = map.pcolormesh(px, py, omega_data[layer_index, :, :], cmap='bwr', vmin=-0.04, vmax=0.04)
    map.colorbar(plotHandle)
    plt.title('f')

    # maxFlux = np.max(np.max(massFlux_lambda), np.max(massFlux_teta))
    # minFlux = np.min(np.min(massFlux_lambda), np.min(massFlux_teta))
    maxFlux = 5*10**-3
    minFlux = -maxFlux

    ax = plt.subplot(223)
    map = Basemap(ax = ax, projection='mill', resolution='l', llcrnrlon=areaRange.lon_min, llcrnrlat=areaRange.lat_min, urcrnrlon=areaRange.lon_max, urcrnrlat=areaRange.lat_max)
    map.drawcoastlines()
    colormeshHandle = map.pcolormesh(px, py, massFlux_lambda, cmap='bwr', vmin=minFlux, vmax=maxFlux)
    map.colorbar(colormeshHandle)
    contourHandle = map.contour(px, py, psi_lambda, linewidths = 3)
    map.colorbar(contourHandle, location='bottom', label='psi contour lines')
    # pl.clim(minFlux, maxFlux)
    plt.title('$m_{\lambda}$')

    ax = plt.subplot(224)
    map = Basemap(ax = ax, projection='mill', resolution='l', llcrnrlon=areaRange.lon_min, llcrnrlat=areaRange.lat_min, urcrnrlon=areaRange.lon_max, urcrnrlat=areaRange.lat_max)
    map.drawcoastlines()
    colormeshHandle = map.pcolormesh(px,py, massFlux_teta, cmap='bwr', vmin=minFlux, vmax=maxFlux)
    map.colorbar(colormeshHandle)
    contourHandle = map.contour(px, py, psi_teta, linewidths = 3)
    map.colorbar(contourHandle, location='bottom', label='psi contour lines')
    # pl.clim(minFlux, maxFlux)
    plt.title('$m_{\\theta}$')
    plt.tight_layout()
    plt.suptitle('diags for layer ' + str(layer_index) + ', p= ' + str(levels_data[layer_index]) + 'hPa', fontsize=24)

for i in range(len(nc_omega_name_list)):
    omega_data_set_vo, zDataSetVO = prepare_data(w_nc_fp_list[i], z_nc_fp_list[i], nc_omega_name_list[i], nc_z_name_list[i], nc_level_name_list[i])

    print 'computing ' + labels_list[i] + ' diags'
    for time_index in range(omega_data_set_vo.fieldData.shape[0]):
        print 'current time_index is ' + str(time_index)
        diags_set_vo = separationController.getLocalHadleyWalkerFluxesProfile(omega_data_set_vo.fieldData[time_index], omega_data_set_vo.lonData, omega_data_set_vo.latData, isXBcsArePeriodic=True, isYBcsAreNeuman=True)

        nc_out_file_path = '/shaheen/project/k1090/osipovs/Temp/circulation_partitioning_into_hadley_walker/' + labels_list[i] + '_' + str(time_index) + '.nc'
        save_data_to_netcdf(nc_out_file_path, omega_data_set_vo.lonData, omega_data_set_vo.latData, omega_data_set_vo.levelsData, omega_data_set_vo.timeData[time_index], diags_set_vo)

        areaRange = AreaRangeVO('Domain', np.min(omega_data_set_vo.latData), np.max(omega_data_set_vo.latData), np.min(omega_data_set_vo.lonData), np.max(omega_data_set_vo.lonData))
        plot_averaged_diags_over_Africa(omega_data_set_vo, zDataSetVO.fieldData[time_index], diags_set_vo)
        save_fig(pics_output_folder, 'hadley walker '  + labels_list[i] + '_' + str(time_index) + '.png')

        layer_index = 5
        plot_individual_layer_diags(omega_data_set_vo.fieldData[time_index], omega_data_set_vo.levelsData, omega_data_set_vo.lonData, omega_data_set_vo.latData, diags_set_vo, layer_index)
        save_fig(pics_output_folder, 'hadley walker ' + labels_list[i] + '_' + str(time_index) + ' single layer.png')

print 'done'

# #check that compute vertical velocities are equal to the era
# omega = omega_lambda+omega_teta
# pl.figure(figsize=(20,10))
# relError = (eraDataSetVO.fieldData[21, :,:]-omega[21,:,:])/eraDataSetVO.fieldData[21,:,:]
# ind = np.abs(eraDataSetVO.fieldData[21,:,:]) > 10**-3
# # levels = np.linspace(-0.5,0.5,11)
# # pl.contourf(relError, levels,  cmap='bwr')
# pl.contourf(relError, cmap='bwr')
# pl.colorbar()
# pl.show()

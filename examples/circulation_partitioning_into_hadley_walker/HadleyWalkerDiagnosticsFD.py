from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as pl
from Projects.HadleyWalker.HadleyWalkerSeparationController import HadleyWalkerSeparationController
from libs.ranges.AreaRangeVO import AreaRangeVO
from libs.readers.DataSetDescriptionVO import DataSetDescriptionVO
from libs.readers.DataSetReadersFactory import DataSetReadersFactory

# Partitioning of the overturning circulaiotn into Hadley & Walker following http://onlinelibrary.wiley.com/doi/10.1002/2013JD020742/epdf

#################
## note that we use here omega in Pa / sec rather tham w in m/sec
#################

descVO = DataSetDescriptionVO(DataSetDescriptionVO.ERA)
# descVO.levelsName = ''
# descVO.variableFilePath = '/home/osipovs/Downloads/netcdf-atls16-a562cefde8a29a7288fa0b8b7f9413f7-SblhfN.nc'
# descVO.variableFilePath = '/home/osipovs/Downloads/netcdf-atls07-a562cefde8a29a7288fa0b8b7f9413f7-0xPUGr.nc'
# descVO.variableFilePath = '/home/osipovs/Downloads/netcdf-atls15-a562cefde8a29a7288fa0b8b7f9413f7-YM55cJ.nc'
descVO.variableFilePath = '/home/osipovs/Data/EraInterim/0.75x0.75/World/1986-1994, World, pressure levels, w, z, montly mean of daily means.nc'
descVO.variableName = 'w'
# descVO.variableName = 'z'
readersFactory = DataSetReadersFactory()
reader = readersFactory.getReader(descVO)
areaRangeVO = AreaRangeVO('belt', -60, 60, 0, 360)
# eraDataSetVO = reader.readDataSet(descVO, areaRangeVO)
eraDataSetVO = reader.readDataSet(descVO)
ind = np.abs(eraDataSetVO.latData) <= 60
eraDataSetVO.latData = eraDataSetVO.latData[ind]
#keep only first time index and values at some single level
# eraDataSetVO.fieldData = eraDataSetVO.fieldData[0,21,ind,:]
eraDataSetVO.fieldData = eraDataSetVO.fieldData[0,:,ind,:]
eraDataSetVO.fieldData = np.rollaxis(eraDataSetVO.fieldData, 1)

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

descVO.variableName = 'z'
zDataSetVO = reader.readDataSet(descVO)
ind = np.abs(zDataSetVO.latData) <= 60
zDataSetVO.latData = zDataSetVO.latData[ind]
zDataSetVO.fieldData = zDataSetVO.fieldData[0,:,ind,:]
zDataSetVO.fieldData = np.rollaxis(zDataSetVO.fieldData, 1)

#get z from geopotential
zDataSetVO.fieldData /= 9.8

separationController = HadleyWalkerSeparationController()
areaRange = AreaRangeVO('Domain', np.min(eraDataSetVO.latData), np.max(eraDataSetVO.latData), np.min(eraDataSetVO.lonData), np.max(eraDataSetVO.lonData))

U, massFlux_lambda, massFlux_teta, omega_lambda, omega_teta, psi_lambda, psi_teta = separationController.getLocalHadleyWalkerFluxesProfile(eraDataSetVO.fieldData, eraDataSetVO.lonData, eraDataSetVO.latData, isXBcsArePeriodic=True, isYBcsAreNeuman=True)

#
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

#extract Hadley and Walker over Africa and average it
maxFlux = 5*10**-3
minFlux = -maxFlux



pl.figure(figsize=(20,10))
ax = pl.subplot(121)
#extract range over Africa only
#because era lon starts from 0 we have to do the logical or
ind = np.logical_or(eraDataSetVO.lonData >= 360-20, eraDataSetVO.lonData <= 60)
currentMassFlux = np.nanmean(massFlux_lambda[:, :, ind], axis=2)
currentOmega = np.nanmean(omega_lambda[:, :, ind], axis=2)

ind = np.logical_or(vDataSetVO.lonData >= 360-20, vDataSetVO.lonData <= 60)
currentV = np.nanmean(vDataSetVO.fieldData[:, :, ind], axis=2)

ind = np.logical_or(zDataSetVO.lonData >= 360-20, zDataSetVO.lonData <= 60)
currentZ = np.nanmean(zDataSetVO.fieldData[:, :, ind], axis=2)
currentZ = np.nanmean(currentZ, axis=1)

currentPsiLambda = np.nanmean(psi_lambda[:, :, ind], axis=2)

pcolorHandle = pl.pcolor(eraDataSetVO.latData, currentZ, currentMassFlux, cmap='bwr', vmin=minFlux, vmax=maxFlux)
pl.colorbar(pcolorHandle)
#pl.quiver(vDataSetVO.latData, vDataSetVO.levelsData, currentV, currentOmega)
#draw zero mass flux contour
# plotHandle = pl.contour(eraDataSetVO.latData, currentZ, currentMassFlux, levels = [0], colors='k',vmin=minFlux, vmax=maxFlux)

#draw psi contours
contourHandle = pl.contour(eraDataSetVO.latData, currentZ, currentPsiLambda, linewidths=3)
pl.colorbar(contourHandle, label='contour lines')

#pl.gca().invert_yaxis()
yLoc = currentZ[::2]
yTicks = vDataSetVO.levelsData[::2]
pl.yticks(yLoc, yTicks)
pl.ylim(currentZ[-1], currentZ[10])
pl.xlabel('latitude')
pl.ylabel('pressure (hPa)')
pl.title('$m_{\lambda}$')
pl.suptitle('Hadley (20E-60W) & Walker (35S-10N)', size='large')

ax = pl.subplot(122)
#extact only 35S to 10 N equatorial belt
ind = np.logical_and(eraDataSetVO.latData >= -35, eraDataSetVO.latData <= 10)
currentData = np.nanmean(massFlux_teta[:, ind, :], axis=1)

ind = np.logical_and(zDataSetVO.latData >= -35, zDataSetVO.latData <= 10)
currentZ = np.nanmean(zDataSetVO.fieldData[:, :, ind], axis=2)
currentZ = np.nanmean(currentZ, axis=1)

currentPsiTeta = np.nanmean(psi_teta[:, ind, :], axis=1)

pcolorHandle = pl.pcolor(eraDataSetVO.lonData, currentZ, currentData, cmap='bwr', vmin=minFlux, vmax=maxFlux)
pl.colorbar(pcolorHandle)
#draw zero mass flux contour
# plotHandle = pl.contour(eraDataSetVO.lonData, currentZ, currentData, levels = [0], colors='k',vmin=minFlux, vmax=maxFlux)

#draw psi contours
contourHandle = pl.contour(eraDataSetVO.lonData, currentZ, currentPsiTeta, linewidths=3)
pl.colorbar(contourHandle, label='contour lines')
yLoc = currentZ[::2]
yTicks = vDataSetVO.levelsData[::2]
pl.yticks(yLoc, yTicks)
pl.ylim(currentZ[-1], currentZ[10])

pl.xlabel('longitude')
pl.ylabel('pressure (hPa)')
pl.title('$m_{\\theta}$')

pl.show()




#lets do single layer
eraDataSetVO.fieldData = eraDataSetVO.fieldData[21,:,:]
U, massFlux_lambda, massFlux_teta, omega_lambda, omega_teta, psi_lambda, psi_teta = separationController.getLocalHadleyWalkerMassFluxes(eraDataSetVO.fieldData, eraDataSetVO.lonData, eraDataSetVO.latData, isXBcsArePeriodic=True, isYBcsAreNeuman=True)

pl.figure(figsize=(20,10))
ax = pl.subplot(2, 2, 1)
map = Basemap(ax = ax, projection='mill', resolution='l', llcrnrlon=areaRange.lon_min, llcrnrlat=areaRange.lat_min, urcrnrlon=areaRange.lon_max, urcrnrlat=areaRange.lat_max)
map.drawcoastlines()
x_m,y_m  = np.meshgrid(eraDataSetVO.lonData,eraDataSetVO.latData)
px, py = map(x_m, y_m)
plotHandle = map.pcolormesh(px, py, U, cmap='bwr')
map.colorbar(plotHandle)
pl.title('poisson solution')

ax = pl.subplot(2, 2, 2)
map = Basemap(ax = ax, projection='mill', resolution='l', llcrnrlon=areaRange.lon_min, llcrnrlat=areaRange.lat_min, urcrnrlon=areaRange.lon_max, urcrnrlat=areaRange.lat_max)
map.drawcoastlines()
plotHandle = map.pcolormesh(px, py, eraDataSetVO.fieldData, cmap='bwr', vmin=-0.4, vmax=0.4)
map.colorbar(plotHandle)
pl.title('f')

# maxFlux = np.max(np.max(massFlux_lambda), np.max(massFlux_teta))
# minFlux = np.min(np.min(massFlux_lambda), np.min(massFlux_teta))
maxFlux = 5*10**-3
minFlux = -maxFlux

ax = pl.subplot(223)
map = Basemap(ax = ax, projection='mill', resolution='l', llcrnrlon=areaRange.lon_min, llcrnrlat=areaRange.lat_min, urcrnrlon=areaRange.lon_max, urcrnrlat=areaRange.lat_max)
map.drawcoastlines()
colormeshHandle = map.pcolormesh(px, py, massFlux_lambda, cmap='bwr', vmin=minFlux, vmax=maxFlux)
map.colorbar(colormeshHandle)
contourHandle = map.contour(px, py, psi_lambda, linewidths = 3)
map.colorbar(contourHandle, location='bottom', label='contour lines')
# pl.clim(minFlux, maxFlux)
pl.title('$m_{\lambda}$')

ax = pl.subplot(224)
map = Basemap(ax = ax, projection='mill', resolution='l', llcrnrlon=areaRange.lon_min, llcrnrlat=areaRange.lat_min, urcrnrlon=areaRange.lon_max, urcrnrlat=areaRange.lat_max)
map.drawcoastlines()
colormeshHandle = map.pcolormesh(px,py, massFlux_teta, cmap='bwr', vmin=minFlux, vmax=maxFlux)
map.colorbar(colormeshHandle)
contourHandle = map.contour(px, py, psi_teta, linewidths = 3)
map.colorbar(contourHandle, location='bottom', label='contour lines')
# pl.clim(minFlux, maxFlux)
pl.title('$m_{\\theta}$')

pl.show()

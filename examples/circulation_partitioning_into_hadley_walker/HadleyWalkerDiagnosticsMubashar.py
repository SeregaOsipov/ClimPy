from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as pl
from Projects.HadleyWalker.HadleyWalkerSeparationController import HadleyWalkerSeparationController
from libs.ranges.AreaRangeVO import AreaRangeVO
from libs.readers.DataSetDescriptionVO import DataSetDescriptionVO
from libs.readers.DataSetReadersFactory import DataSetReadersFactory

#read omega data set
descVO = DataSetDescriptionVO(DataSetDescriptionVO.GENERIC_NETCDF)
descVO.latName = 'lat'
descVO.lonName = 'lon'
descVO.variableFilePath = '/home/osipovs/ExternalStorages/Shaheen/project/k14/DATA_MUBAHSAR/HIRAM/omega/omega_hiram1_JJA_TotalPeriod_Climatology.nc'
descVO.variableName = 'omega'

readersFactory = DataSetReadersFactory()
reader = readersFactory.getReader(descVO)
areaRangeVO = AreaRangeVO('belt', -60, 60, 0, 360)
omegaDataSetVO = reader.readDataSet(descVO)
ind = np.abs(omegaDataSetVO.latData) <= 60
omegaDataSetVO.latData = omegaDataSetVO.latData[ind]
#select the time index and data in the lat band
omegaDataSetVO.fieldData = omegaDataSetVO.fieldData[0,:,ind,:]
omegaDataSetVO.fieldData = np.rollaxis(omegaDataSetVO.fieldData, 1)

#read z data set
descVO.variableFilePath = '/home/osipovs/ExternalStorages/Shaheen/project/k14/DATA_MUBAHSAR/HIRAM/geop_height/hght_hiram1_JJA_TotalPeriod_Climatology.nc'
descVO.variableName = 'hght'
zDataSetVO = reader.readDataSet(descVO)
ind = np.abs(zDataSetVO.latData) <= 60
zDataSetVO.latData = zDataSetVO.latData[ind]
zDataSetVO.fieldData = zDataSetVO.fieldData[0,:,ind,:]
zDataSetVO.fieldData = np.rollaxis(zDataSetVO.fieldData, 1)

separationController = HadleyWalkerSeparationController()
areaRange = AreaRangeVO('Domain', np.min(omegaDataSetVO.latData), np.max(omegaDataSetVO.latData), np.min(omegaDataSetVO.lonData), np.max(omegaDataSetVO.lonData))

#extract Hadley and Walker over Africa and average it
U, massFlux_lambda, massFlux_teta, omega_lambda, omega_teta, psi_lambda, psi_teta = separationController.getLocalHadleyWalkerFluxesProfile(omegaDataSetVO.fieldData, omegaDataSetVO.lonData, omegaDataSetVO.latData, isXBcsArePeriodic=True, isYBcsAreNeuman=True)

#flux range for the plots
maxFlux = 5*10**-3
minFlux = -maxFlux


pl.figure(figsize=(20,10))
ax = pl.subplot(121)
#extract range over Africa only
#because lon starts from 0 we have to do the logical or
ind = np.logical_or(omegaDataSetVO.lonData >= 360-20, omegaDataSetVO.lonData <= 60)
currentMassFlux = np.nanmean(massFlux_lambda[:, :, ind], axis=2)
currentOmega = np.nanmean(omega_lambda[:, :, ind], axis=2)

ind = np.logical_or(zDataSetVO.lonData >= 360-20, zDataSetVO.lonData <= 60)
currentZ = np.nanmean(zDataSetVO.fieldData[:, :, ind], axis=2)
currentZ = np.nanmean(currentZ, axis=1)

currentPsiLambda = np.nanmean(psi_lambda[:, :, ind], axis=2)

pcolorHandle = pl.pcolor(omegaDataSetVO.latData, currentZ, currentMassFlux, cmap='bwr', vmin=minFlux, vmax=maxFlux)
pl.colorbar(pcolorHandle)
#pl.quiver(vDataSetVO.latData, vDataSetVO.levelsData, currentV, currentOmega)
#draw zero mass flux contour
plotHandle = pl.contour(omegaDataSetVO.latData, currentZ, currentMassFlux, levels = [0], colors='k',vmin=minFlux, vmax=maxFlux)

#draw psi contours
# contourHandle = pl.contour(omegaDataSetVO.latData, currentZ, currentPsiLambda, linewidths=3)
# pl.colorbar(contourHandle, label='contour lines')

#pl.gca().invert_yaxis()
yLoc = currentZ[::2]
yTicks = omegaDataSetVO.levelsData[::2]
pl.yticks(yLoc, yTicks)
pl.ylim(currentZ[0], currentZ[11])
pl.xlabel('latitude')
pl.ylabel('pressure (hPa)')
pl.title('$m_{\lambda}$')
pl.suptitle('Hadley (20E-60W) & Walker (35S-10N)', size='large')

ax = pl.subplot(122)
#extact only 35S to 10 N equatorial belt
ind = np.logical_and(omegaDataSetVO.latData >= -35, omegaDataSetVO.latData <= 10)
currentData = np.nanmean(massFlux_teta[:, ind, :], axis=1)

ind = np.logical_and(zDataSetVO.latData >= -35, zDataSetVO.latData <= 10)
currentZ = np.nanmean(zDataSetVO.fieldData[:, :, ind], axis=2)
currentZ = np.nanmean(currentZ, axis=1)

currentPsiTeta = np.nanmean(psi_teta[:, ind, :], axis=1)

pcolorHandle = pl.pcolor(omegaDataSetVO.lonData, currentZ, currentData, cmap='bwr', vmin=minFlux, vmax=maxFlux)
pl.colorbar(pcolorHandle)
#draw zero mass flux contour
plotHandle = pl.contour(omegaDataSetVO.lonData, currentZ, currentData, levels = [0], colors='k',vmin=minFlux, vmax=maxFlux)

#draw psi contours
# contourHandle = pl.contour(omegaDataSetVO.lonData, currentZ, currentPsiTeta, linewidths=3)
# pl.colorbar(contourHandle, label='contour lines')
yLoc = currentZ[::2]
yTicks = omegaDataSetVO.levelsData[::2]
pl.yticks(yLoc, yTicks)
pl.ylim(currentZ[0], currentZ[11])

pl.xlabel('longitude')
pl.ylabel('pressure (hPa)')
pl.title('$m_{\\theta}$')

pl.show()




## lets do single layer, i.e. LOCAL Hadley and Walker circulation
#pick a layer at around 500 hPa
omegaDataSetVO.fieldData = omegaDataSetVO.fieldData[6,:,:]
U, massFlux_lambda, massFlux_teta, omega_lambda, omega_teta, psi_lambda, psi_teta = separationController.getLocalHadleyWalkerMassFluxes(omegaDataSetVO.fieldData, omegaDataSetVO.lonData, omegaDataSetVO.latData, isXBcsArePeriodic=True, isYBcsAreNeuman=True)

pl.figure(figsize=(20,10))
ax = pl.subplot(2, 2, 1)
map = Basemap(ax = ax, projection='mill', resolution='l', llcrnrlon=areaRange.lon_min, llcrnrlat=areaRange.lat_min, urcrnrlon=areaRange.lon_max, urcrnrlat=areaRange.lat_max)
map.drawcoastlines()
x_m,y_m  = np.meshgrid(omegaDataSetVO.lonData,omegaDataSetVO.latData)
px, py = map(x_m, y_m)
plotHandle = map.pcolormesh(px, py, U, cmap='bwr')
map.colorbar(plotHandle)
pl.title('poisson solution')

ax = pl.subplot(2, 2, 2)
map = Basemap(ax = ax, projection='mill', resolution='l', llcrnrlon=areaRange.lon_min, llcrnrlat=areaRange.lat_min, urcrnrlon=areaRange.lon_max, urcrnrlat=areaRange.lat_max)
map.drawcoastlines()
plotHandle = map.pcolormesh(px, py, omegaDataSetVO.fieldData, cmap='bwr', vmin=-0.4, vmax=0.4)
map.colorbar(plotHandle)
pl.title('forcing, ie omega')

maxFlux = 5*10**-3
minFlux = -maxFlux

ax = pl.subplot(223)
map = Basemap(ax = ax, projection='mill', resolution='l', llcrnrlon=areaRange.lon_min, llcrnrlat=areaRange.lat_min, urcrnrlon=areaRange.lon_max, urcrnrlat=areaRange.lat_max)
map.drawcoastlines()
colormeshHandle = map.pcolormesh(px, py, massFlux_lambda, cmap='bwr', vmin=minFlux, vmax=maxFlux)
map.colorbar(colormeshHandle)
# contourHandle = map.contour(px, py, psi_lambda, linewidths = 3)
# map.colorbar(contourHandle, location='bottom', label='contour lines')
pl.title('$m_{\lambda}$')

ax = pl.subplot(224)
map = Basemap(ax = ax, projection='mill', resolution='l', llcrnrlon=areaRange.lon_min, llcrnrlat=areaRange.lat_min, urcrnrlon=areaRange.lon_max, urcrnrlat=areaRange.lat_max)
map.drawcoastlines()
colormeshHandle = map.pcolormesh(px,py, massFlux_teta, cmap='bwr', vmin=minFlux, vmax=maxFlux)
map.colorbar(colormeshHandle)
# contourHandle = map.contour(px, py, psi_teta, linewidths = 3)
# map.colorbar(contourHandle, location='bottom', label='contour lines')
pl.title('$m_{\\theta}$')

pl.show()

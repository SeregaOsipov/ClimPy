import datetime as dt
import os as os
import numpy as np
from pyhdf.SD import SD, SDC

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'


def get_modis_montly_file_paths(year, terra=True):
    '''
    Build files paths to the monthly MODIS data
    '''
    fp_root = '/work/mm0062/b302074/Data/NASA/MYD08_M3/'  # Aqua
    if terra:
        fp_root = '/work/mm0062/b302074/Data/NASA/MOD08_M3/'  # Terra

    monthly_fps = []
    for month in range(1, 13):
        requestedDate = dt.date(year, month, 1)
        yday = (requestedDate - dt.date(requestedDate.year, 1, 1)).days + 1
        # print(yday)
        folder_path = fp_root + str(year) + '/' + str(yday).zfill(3) + '/'
        files = os.listdir(folder_path)
        if len(files) > 1:
            raise Exception("multiple files!")
        fp = folder_path + files[0]
        monthly_fps.append(fp)

    return monthly_fps


def get_modis_var(fp, key):
    hdf = SD(fp, SDC.READ)

    lat = hdf.select('YDim')[:]
    lon = hdf.select('XDim')[:]
    aod_var = hdf.select(key)

    scale = aod_var.attributes()['scale_factor']
    offset = aod_var.attributes()['add_offset']
    fill_value = aod_var.attributes()['_FillValue']

    aod = aod_var[:].astype(float)
    aod[aod == fill_value] = np.NaN
    aod = aod * scale + offset

    # timeVariable = hdf.select('Scan_Start_Time')
    # timeData = netCDF4.num2date(timeVariable[:],timeVariable.units)
    # timeData = timeData[0,0]

    hdf.end()

    vo = {}
    vo[key] = aod
    vo['lat'] = lat
    vo['lon'] = lon

    return vo


def prepare_modis_aod_climatology(key):
    """
    read and prepare modis aod data and climatology
    :param key: 'AOD_550_Dark_Target_Deep_Blue_Combined_Mean_Mean' or  'Aerosol_Optical_Depth_Small_Ocean_Mean_Mean'
    :return:
    """

    modis_aod_data = np.zeros((180, 360, 12, 5))
    for year in range(2008, 2012):
        fps = get_modis_montly_file_paths(year)
        for month in range(1, 13):
            fp = fps[month-1]
            vo = get_modis_var(fp, key)
            modis_aod_data[:, :, month - 1, year - 2008] += vo[key]

    # remove negative values from MODIS data
    modis_aod_data[modis_aod_data < 0] = 0
    modis_aod_clim = np.nanmean(modis_aod_data, axis=3)

    modis_aod_vo = {}
    modis_aod_vo['lat'] = vo['lat']
    modis_aod_vo['lon'] = vo['lon']

    modis_aod_vo[key] = modis_aod_data
    modis_aod_vo['aod_clim'] = modis_aod_clim

    return modis_aod_vo
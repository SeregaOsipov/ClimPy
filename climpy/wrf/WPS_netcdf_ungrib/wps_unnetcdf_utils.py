import struct
import datetime as dt
import netCDF4
import numpy as np
import sys

__author__ = 'Sergey Osipov <Serheadega.Osipov@gmail.com>'


##########################
# Use the METGRID.TBL var names as a key for the dictionary that holds the mapping rules for this variable
# Use the METGRID.TBL key as primary
# otherwise add custom field to the METGRID.TBL and link it to the existing (such as U and UU)
# the mapping is METGRID.TBL field <-> the MERRA2/ECMWF/etc NETCDF var post-processing rules
##########################

# //TODO: fix the mapping, it should be METGRID.TBL <-> NETCDF var name (ECMWF)
# The mapping from ECMWF to WRF
_FIELD_MAP_ECMWF_OA_2_WRF = {
                #SURFACE variables
                'z' : 'SOILGEO', #129
                'lsm' : 'LANDSEA',
                'sp' : 'PSFC',
                'sd' : 'SNOW_EC', #141
                'msl' : 'PMSL',
                'u10' : 'UU',
                'v10' : 'VV',
                't2m' : 'TT',
                'd2m' : 'DEWPT',
                'skt' : 'SKINTEMP',
                'ci' : 'SEAICE', #31
                #'rsn' : 'rsn', #33, snow density, not in the Vtable, for pp
                'sst' : 'SST', #34

                'swvl1' : 'SM000007', #39
                'swvl2' : 'SM007028',
                'swvl3' : 'SM028100',
                'swvl4' : 'SM100289',

                'stl1' : 'ST000007', #139
                'stl2' : 'ST007028', #170
                'stl3' : 'ST028100', #183
                'stl4' : 'ST100289', #236
                'stl1' : 'ST000007',

                #MODEL LEVEL variables
                't' : 'TT',
                'q' : 'SPECHUMD',
                'u' : 'UU',
                'v' : 'VV',

                # auxiliry variables, not saved, but used
                'latitude': 'latitude',
                'longitude': 'longitude',
                'time': 'time',
                'level': 'level',
                }

# The mapping for MERRA2
# the post-processing for this maps includes:
# 0. PRECOMPUTE pressure to avoid calc_emcwf_p.exe
# 1. Check that it is OK to mix 3D RH and 2m specific humidity

#######################################
# Below you will find a bunch of the post-processing rules and auxiliary utils
#######################################


class WrfMetgridMapItem:
    def __init__(self, netcdf_var_key, at_sea_level_pressure=None, pp_impl=None, read_in_pp_impl=False, override_time=False):
        self.netcdf_var_key = netcdf_var_key
        self.at_sea_level_pressure = at_sea_level_pressure  # this will force xlvl to 201300
        self.pp_impl = pp_impl  # function to carry out any necessary post-processing, unique for each dataset (MERRA2)
        # this flag indicates that reading of the netcdf variables should be delayed until the pp_impl
        self.read_in_pp_impl = read_in_pp_impl
        # use this flag for time invariant fields which may have "wrong" date
        self.override_time = override_time


def invert_mask(nc_data, nc, time_index, level_index):
    '''
    Invert 0/1 mask
    '''
    nc_data['slab'] = 1 - nc_data['slab']


def divide_by_cos_lat(nc_data, nc, time_index, level_index):
    # nc_data['slab'] /= np.cos(np.deg2rad(nc.variables['lat'][:))
    if 'time' in nc.variables['coslat'].dimensions:
        nc_data['slab'] /= nc.variables['coslat'][time_index]
    else:
        nc_data['slab'] /= nc.variables['coslat'][:]  # only lat lon


def convert_geopotential_to_height(nc_data, nc, time_index, level_index):
    nc_data['slab'] /= 9.81
    nc_data['units'] = 'm'


def potential_to_regular_temperature(nc_data, nc, time_index, level_index):
        pressure = nc.variables['press'][time_index, level_index]  # Pa
        # convert it to normal temperature: Tn = Tp / (p0/p)^kappa
        kappa = 0.2854
        nc_data['slab'] /= (1000.*10**2/pressure)**kappa


def convert_unit_ratio_to_percents(nc_data, nc, time_index, level_index):
    nc_data['slab'] *= 100
    nc_data['units'] = '%'


def derive_land_sea_merra2(nc_data, nc, time_index, level_index):
    water_fraction = nc.variables['FROCEAN'][time_index] + nc.variables['FRLAKE'][time_index]
    nc_data['slab'] = np.zeros(water_fraction.shape)
    ind = water_fraction < 0.5
    nc_data['slab'][ind] = 1


def derive_3d_pressure_merra2(nc_data, nc, time_index, level_index):
    # To get the pressure for a selected layer I still have to build the entire 3d field first
    # TODO: use derive_merra2_pressure_stag_profile from merra_utils.py
    # see this doc for details on Vertical Structure https://gmao.gsfc.nasa.gov/pubs/docs/Bosilovich785.pdf
    layer_pressure_thickness = nc.variables['DELP'][time_index]  # 3d

    # summation has to start from the top, p_top is fixed to 1 Pa
    pressure_stag_no_toa = 1 + np.cumsum(layer_pressure_thickness, axis=0)
    # new shape
    stag_shape = (pressure_stag_no_toa.shape[0]+1,) + pressure_stag_no_toa.shape[1:]
    pressure_stag = np.empty(stag_shape)
    pressure_stag[0, :, :] = 1  # fixed p_top
    pressure_stag[1:, :, :] = pressure_stag_no_toa

    # get the pressure at the rho grid
    pressure_rho = (pressure_stag[1:] + pressure_stag[:-1])/2
    nc_data['slab'] = pressure_rho[level_index]


def interpolate_soil_temperatures_merra2(nc_data, nc, time_index, level_index):

    # //TODO: this function is not implemented yet

    # //TODO: this function will be executed every time, think how to accelerate it if performance is poor
    # derive the MERRA2 soil vertical grid
    lnd_const_nc_file_path = get_merra2_file_path('const_2d_lnd_Nx', dt.datetime(1, 1, 1))
    lnd_const_nc = netCDF4.Dataset(lnd_const_nc_file_path)

    merra_soil_layer_thickness = np.empty((6,) + lnd_const_nc.variables['dzgt1'].shape[1:])  # m
    merra_soil_layer_thickness[0] = lnd_const_nc.variables['dzgt1'][0]
    merra_soil_layer_thickness[1] = lnd_const_nc.variables['dzgt2'][0]
    merra_soil_layer_thickness[2] = lnd_const_nc.variables['dzgt3'][0]
    merra_soil_layer_thickness[3] = lnd_const_nc.variables['dzgt4'][0]
    merra_soil_layer_thickness[4] = lnd_const_nc.variables['dzgt5'][0]
    merra_soil_layer_thickness[5] = lnd_const_nc.variables['dzgt6'][0]

    soil_z_stag_no_surf = np.cumsum(merra_soil_layer_thickness, axis=0)
    # Note that the vertical profile of temperature represented by TSOIL1 through TSOIL6 should be “shifted downward” by DZTS.
    dzts = lnd_const_nc.variables['dzts'][0]
    soil_z_stag_no_surf += dzts

    # convert m -> cm
    soil_z_stag_no_surf *= 100

    #include the surface layer
    stag_shape = (soil_z_stag_no_surf.shape[0] + 1,) + soil_z_stag_no_surf.shape[1:]
    soil_z_stag = np.empty(stag_shape)
    soil_z_stag[0] = 0
    soil_z_stag[1:] = soil_z_stag_no_surf
    soil_z_rho = (soil_z_stag[:-1]+soil_z_stag[1:])/2

    import matplotlib.pyplot as plt
    plt.contourf(merra_soil_layer_thickness[0])
    plt.colorbar()

    plt.clf()
    plt.plot(soil_z_stag[:,150,200], '-o')
    plt.yscale('log')

    soil_z_rho[:,150,200]
    #the edges are 0, 10, 30, 68, 144, 295, 1295 cm
    soil_z_stag[:, 150, 200]


def split_soil_temperature(nc_data, nc, time_index, level_index):
    '''
    EMAC specific implementation
    This breaks 3d deep soil temperature for each layer slice

    Layers are:
    fill_lev =   3 : ST000003(200100)
    fill_lev =  19 : ST003019(200100)
    fill_lev =  78 : ST019078(200100)
    fill_lev = 268 : ST078268(200100)
    fill_lev = 698 : ST268698(200100)

    :param nc_data:
    :param nc:
    :param time_index:
    :param level_index:
    :return:
    '''

    # determine level by field
    soil_layer = 0
    if nc_data['field'] == 'ST003019':
        soil_layer = 1
    if nc_data['field'] == 'ST019078':
        soil_layer = 2
    if nc_data['field'] == 'ST078268':
        soil_layer = 3
    if nc_data['field'] == 'ST268698':
        soil_layer = 4

    nc_data['slab'] = nc_data['slab'][soil_layer]


def split_soil_moisture(nc_data, nc, time_index, level_index):
    '''
    EMAC specific implementation

    Below is the explanation from Klaus:

    Hello Sergey
    ws is the water volume per area. To convert to volumetric soil moisture, divide by the rooting depth, representing the depth of the
    soil column available for moisture (and plant roots). On mistral you can find a rooting depth dataset here:
    /pool/data/MESSY/DATA/MESSy2/raw/onemis/Astitha/RootDepth1deg_2002.nc
    This will probably result in some values larger than 1, so make sure to apply an upper limit.
    I hope this helps!
    Best wishes
    Klaus

    Layers are:
    fill_lev =   3 : SM000003(200100)
    fill_lev =  19 : SM003019(200100)
    fill_lev =  78 : SM019078(200100)
    fill_lev = 268 : SM078268(200100)
    fill_lev = 698 : SM268698(200100)

    :param nc_data:
    :param nc:
    :param time_index:
    :param level_index:
    :return:
    '''

    # determine level by field
    soil_layer = 0
    if nc_data['field'] == 'SM003019':
        soil_layer = 1
    if nc_data['field'] == 'SM019078':
        soil_layer = 2
    if nc_data['field'] == 'SM078268':
        soil_layer = 3
    if nc_data['field'] == 'SM268698':
        soil_layer = 4

    layer_depth = float(nc_data['field'][5:])/100  # m

    # I made a local copy
    # rooting_depth_file_path = '/work/mm0062/b302074/Data/AirQuality/AQABA/CMIP6/RootDepth1deg_2002.nc'
    # rooting_depth_file_path = '/work/mm0062/b302074/Data/AirQuality/AQABA/CMIP6/EMAC_auxilary/RootDepthT42_2002.nc'
    # TODO: Andrea now outputs rooting depth in EMAC, read via aux channel. But, land model in EMAC is so bad, just wait until so come up with a better one
    # this is the original file which is regrided by EMAC in runtime '/pool/data/MESSY/DATA/MESSy2/raw/onemis/GSDT_0.3_X_X_RootDepth_2000.nc'
    # The data is supposed to be static and interpolated on the EMAC grid
    rooting_depth_file_path = '/work/mm0062/b302011/script/Osipov/simulations/AQABA_2017/MIM_STD________20170701_0000_WRF_bc.nc'   # this needs to be enabled in the EMAC regrider output
    rd_nc = netCDF4.Dataset(rooting_depth_file_path)
    # rooting_depth = rd_nc.variables['DEPTH'][0, 0]  # m , source file has lev dimensions with size 1
    rooting_depth = rd_nc.variables['rdepth_root_depth'][0]  # m
    rooting_depth[rooting_depth==-1] = np.NaN  # sea mask

    # TODO: until the file is regridded assume rooting depth is 1 m
    # rooting_depth = np.ones(nc_data['slab'].shape)
    # TODO: have to setup realistic profile, for now uniform above rooting depth
    # deeper than rooting depth, then set 0
    ind = rooting_depth > layer_depth
    nc_data['slab'][ind] /= rooting_depth[ind]
    nc_data['slab'][np.logical_not(ind)] = 0

    # filter unrealistic values
    ind = nc_data['slab'] > 1
    nc_data['slab'][ind] = 1

    # update units
    nc_data['units'] = 'm-3 m-3'


# //TODO: MERRA2 profiles do not correspond to the METGRID.TBL vertical grid, have to interpolate
_FIELD_MAP_MERRA_2_WRF = {
    # inst1_2d_asm_Nx, SURFACE variables
    'PSFC': WrfMetgridMapItem('PS'),  # surface pressure, Pa
    'PMSL': WrfMetgridMapItem('SLP', at_sea_level_pressure=True),  # mean sea level pressure, Pa

    # UU and VV keys are taken by the 3d field, use U and V as a substitute for the 10m winds
    'U': WrfMetgridMapItem('U10M'),  # U at 10 m
    'V': WrfMetgridMapItem('V10M'),  # V at 10 m
    'T': WrfMetgridMapItem('T2M'),  # Temperature at 2 m, K
    'SPECHUMD': WrfMetgridMapItem('QV2M'),  # 2-meter_specific_humidity, kg kg**-1
    'SKINTEMP': WrfMetgridMapItem('TS'),  # skin temperature, K


    # tavg1_2d_ocn_Nx
    'SEAICE': WrfMetgridMapItem('FRSEAICE'), #ice covererd fraction of tile
    # 'sst' : WrfMetgridMapItem('SST'), //TODO: check how SST is prescribed, is it skin temperature?


    # const_2d_asm_Nx
    'SOILHGT': WrfMetgridMapItem('PHIS', pp_impl=convert_geopotential_to_height, override_time=True),  # surface geopotential height, m
    'LANDSEA': WrfMetgridMapItem('FROCEAN', pp_impl=derive_land_sea_merra2, read_in_pp_impl=True, override_time=True),  # fraction_of_ocean


    # tavg1_2d_lnd_Nx, land variables
    # In this approach I've updated the METGRID.TBL and inserted custom depths
    'ST000010': WrfMetgridMapItem('TSOIL1'),  # soil temperatures in layers 1-5, K
    'ST010030': WrfMetgridMapItem('TSOIL2'),
    'ST030068': WrfMetgridMapItem('TSOIL3'),
    'ST068144': WrfMetgridMapItem('TSOIL4'),
    'ST144295': WrfMetgridMapItem('TSOIL5'),

    # if custom (MERRA2) depths do not work, you can do interpolation by hand
    # 'ST000010': WrfMetgridMapItem('TSOIL1', pp_impl=interpolate_soil_temperatures_merra2, read_in_pp_impl=True),  # soil temperatures layer 1-5, K
    # 'ST010020': WrfMetgridMapItem('TSOIL2', pp_impl=interpolate_soil_temperatures_merra2, read_in_pp_impl=True),
    # 'ST020040': WrfMetgridMapItem('TSOIL3', pp_impl=interpolate_soil_temperatures_merra2, read_in_pp_impl=True),
    # 'ST040080': WrfMetgridMapItem('TSOIL4', pp_impl=interpolate_soil_temperatures_merra2, read_in_pp_impl=True),
    # 'ST080150': WrfMetgridMapItem('TSOIL5', pp_impl=interpolate_soil_temperatures_merra2, read_in_pp_impl=True),


    # The surface soil moisture (SFMC andGWETTOP) is the average soil moisture for the top DZSF=0.02 m of the soil.
    # The root zone soil moisture variable, RZMC, ostensibly refers to the average amount of water ina nominal “root zone” of 1 meter depth
    #'SM000002': WrfMetgridMapItem('SFMC'),  # surface soil moisture (level 1), m**-3 m**-3
    #'SM000100': WrfMetgridMapItem('RZMC'),  # water root zone

    # WRF treats soil temperature and humidity on the same vertical grid (see SOIL_LAYERS in the METGRID.TBL)
    # Since MERRA2 has only 2 levels in the output, I decided to do coarse interpolation by hand
    'SM000010': WrfMetgridMapItem('SFMC'),  # surface soil moisture (level 1), m**-3 m**-3
    'SM010030': WrfMetgridMapItem('RZMC'),  # water root zone
    'SM030068': WrfMetgridMapItem('RZMC'),  # extrapolate values
    'SM068144': WrfMetgridMapItem('RZMC'),
    'SM144295': WrfMetgridMapItem('RZMC'),

    'SNOWH': WrfMetgridMapItem('SNODP'),  # Physical Snow Depth, m
    'SNOW': WrfMetgridMapItem('SNOMAS'),  # Water equivalent snow depth, kg m**-2


    # inst3_3d_asm_Nv, model level 3D variables
    'PRESSURE': WrfMetgridMapItem('DELP', pp_impl=derive_3d_pressure_merra2, read_in_pp_impl=True),  # Pressure, Pa
    'HGT': WrfMetgridMapItem('H'),  # Height (rho grid), m
    'TT': WrfMetgridMapItem('T'),  # Temperature, K
    'RH': WrfMetgridMapItem('RH', pp_impl=convert_unit_ratio_to_percents),  # relative humidity, %
    'UU': WrfMetgridMapItem('U'),
    'VV': WrfMetgridMapItem('V'),


    # auxiliary translation of the variables and dimensions
    # they are axillary and are not exported to the intermediate format
    # I may need to introduce vars and dims separately
    'latitude': 'lat',
    'longitude': 'lon',
    # 'time' : 'time',
    'level': 'lev',
    }


LATLON_PROJECTION = 0
GAUSSIAN_PROJECTION = 4


def format_hdate(time_data):
    return time_data.strftime('%Y-%m-%d_%H:%M:%S')  # format as "2008:01:01_00:00:00"


def prepare_nc_data(nc, _FIELD_MAP, var_key, time_index, level_index, map_projection_version):
    # map_projection_version possible values:
    # 0 is lat-lon projection (Cylindrical equidistant)
    # 4 is Gaussian projections
    # inspect metgrid/src/read_met_module.F90 & write_met_module.F90 for details

    nc_var_key = _FIELD_MAP[var_key].netcdf_var_key

    nc_data = {}
    nc_data['nx'] = nc.dimensions[_FIELD_MAP['longitude']].size
    nc_data['ny'] = nc.dimensions[_FIELD_MAP['latitude']].size
    nc_data['units'] = ''
    if hasattr(nc.variables[nc_var_key], 'units'):
        nc_data['units'] = nc.variables[nc_var_key].units

    lons = nc.variables[_FIELD_MAP['longitude']][:]
    lats = nc.variables[_FIELD_MAP['latitude']][:]

    # I have to specify south west corner of the data, adjust data accordingly
    # TODO: print('Flip_lat_dim is probably dataset dependent, check carefully')
    flip_lat_dim = False
    nc_data['startlat'] = lats[0]
    nc_data['deltalat'] = (lats[-1] - lats[0]) / (nc_data['ny'] - 1)
    if lats[0] > lats[1]:
        flip_lat_dim = True
        nc_data['startlat'] = lats[-1]
        nc_data['deltalat'] = (lats[0] - lats[-1]) / (nc_data['ny'] - 1)

    nc_data['startlon'] = lons[0]
    nc_data['deltalon'] = (lons[-1] - lons[0]) / (nc_data['nx'] - 1)

    # ! Gaussian projection (iproj=4) parameters
    # !nx, ny, xlvl     ! Vertical level of data in 2 - d array
    # !field   ! Name of the field
    # !startloc !"SWCORNER" for Gaussian projection
    # !startlat, startlon, nlats, deltalon, earth_radius
    # !hdate, xfcst, map_source

    nc_data['version'] = 5  # data format version, 5 means WPS
    nc_data['iproj'] = map_projection_version

    # //TODO: the set of variables depends on the projection

    nc_data['xlvl'] = 200100  # indicates surface data
    if _FIELD_MAP['level'] in nc.variables[nc_var_key].dimensions:  # nc.dimensions.keys():
        nc_data['xlvl'] = nc.variables[_FIELD_MAP['level']][level_index]

    if _FIELD_MAP[var_key].at_sea_level_pressure:
        nc_data['xlvl'] = 201300  # indicates sea-level pressure

    nc_data['field'] = var_key  # remember that the var key has to be as in the METGRID.TBL
    nc_data['startloc'] = "SWCORNER"  # specific for Gaussian projection, Which point in array is given by startlat/startlon

    nc_data['nlats'] = nc_data['ny'] / 2  # // TODO: not sure this is correct
    nc_data['earth_radius'] = 6367.47  # the value is taken from the grib reader # ml data reported value 6371.229492

    nc_data['time_data'] = netCDF4.num2date(nc.variables['time'][time_index], nc.variables['time'].units)  # TODO: these dates are ugly
    nc_data['hdate'] = format_hdate(nc_data['time_data'])
    nc_data['xfcst'] = 0.0
    nc_data['map_source'] = "EMAC CCMI"  # seems to be arbitrary

    # !nc_data % dx = 5.0
    # !nc_data % dy = 5.0
    # !nc_data % xlonc = 0.0
    # !nc_data % truelat1 = 0.0
    # !nc_data % truelat2 = 0.0

    nc_data['is_wind_grid_rel'] = False  # //TODO: check this for each new data set
    # nc_data['desc'] = nc.variables[nc_var_key].long_name
    nc_data['desc'] = ''

    # read the slab data if it based only on a single netcdf variable
    if not _FIELD_MAP[var_key].read_in_pp_impl:
        if _FIELD_MAP['level'] in nc.variables[nc_var_key].dimensions:
            nc_data['slab'] = nc.variables[nc_var_key][time_index, level_index]
        else:
            nc_data['slab'] = nc.variables[nc_var_key][time_index]

    # if necessary, do the post-processing now
    if _FIELD_MAP[var_key].pp_impl:
        print('doing pp for {}'.format(var_key))
        _FIELD_MAP[var_key].pp_impl(nc_data, nc, time_index, level_index)

    # has to flip the data
    if flip_lat_dim:
        nc_data['slab'] = np.flip(nc_data['slab'], axis=0)

    # fill the missing values according to the METGRID.TBL
    if type(nc_data['slab']) is np.ma.MaskedArray:
        missing_value = -1.E30
        nc_data['slab'] = nc_data['slab'].filled(missing_value)
        # print(var_key + ' has missing values, filling them with {}'.format(missing_value))

    # //TODO: code below is for EMCWF only and needs to be replaced with the pp_impl logic

    # reprocessing SURFACE field, see WPS rrpr.f
    if var_key == 'z':
        nc_data['slab'] /= 9.81
        nc_data['units'] = 'm'
        nc_data['field'] = 'SOILHGT'
    elif var_key == 'd2m':
         Xlv = 2.5e6
         Rv = 461.5
         # we already have the dew temperature, read the regular temperature
         T = nc.variables['t2m'][time_index]
         dp = nc_data['slab']
         rh = np.exp( Xlv / Rv * (1. / T - 1. / dp)) * 1.E2

         nc_data['slab'] = rh
         nc_data['units'] = '%'
         nc_data['field'] = 'RH'
    elif var_key == 'sd':
        # nc_data['slab'] *= 1000
        # looks like ECMWF has snow density which is not equal to the 1000, use the variable
        snow_density_data = nc.variables['rsn'][time_index]
        nc_data['slab'] *= snow_density_data
        nc_data['units'] = 'kg m-2'
        nc_data['field'] = 'SNOW'
    elif var_key == 'lsm' or var_key == 'ci':  # for SEAICE and LANDSEA convert fraction to binary 0 1
        ind = nc_data['slab'] > 0.5
        nc_data['slab'][ind] = 1
        nc_data['slab'][np.logical_not(ind)] = 0
        nc_data['units'] = '0/1 Flag'

    return nc_data


def wrf_write(f, nc_data):
    # f = open(out_file_name, 'wb')

    # determine the endianness (little or big)
    endian_fmt = '<'  # little-endian
    if sys.byteorder == 'big':
        endian_fmt = '>'
    # print('endian fmt is {}'.format(endian_fmt))
    # looks like WRF forces the big endian in the unformatted files (compiler flag -convert big_endian)
    endian_fmt = '>'

    string_encoding = 'ascii'  # utf-8
    # > in the format means big endian
    # each record (not variable) is surrounded by record length

    # record 1
    # looks like I need to write the record length
    f.write(struct.pack(endian_fmt+'i', 4))
    f.write(struct.pack(endian_fmt+'i', nc_data['version']))
    f.write(struct.pack(endian_fmt+'i', 4))

    if (nc_data['iproj'] == 0 or nc_data['iproj'] == 4): #lat-lon and Gaussian have the same of variables
        # record 2 , fmt = endian_fmt+"24sf32s9s25s46sfiii"

        record_size = struct.calcsize(endian_fmt+"24sf32s9s25s46sfiii")
        f.write(struct.pack(endian_fmt+'i', record_size))

        formatted_string = '{0:24s}'.format(nc_data['hdate'])
        f.write(struct.pack(endian_fmt+'24s', formatted_string.encode(string_encoding)))

        f.write(struct.pack(endian_fmt+'f', nc_data['xfcst']))

        formatted_string = '{0:32s}'.format(nc_data['map_source'])
        f.write(struct.pack(endian_fmt+'32s', formatted_string.encode(string_encoding)))

        formatted_string = '{0:9s}'.format(nc_data['field'])
        f.write(struct.pack(endian_fmt+'9s', formatted_string.encode(string_encoding)))

        formatted_string = '{0:25s}'.format(nc_data['units'])
        f.write(struct.pack(endian_fmt+'25s', formatted_string.encode(string_encoding)))

        formatted_string = '{0:46s}'.format(nc_data['desc'])
        f.write(struct.pack(endian_fmt+'46s', formatted_string.encode(string_encoding)))

        f.write(struct.pack(endian_fmt+'f', nc_data['xlvl']))
        f.write(struct.pack(endian_fmt+'i', nc_data['nx']))
        f.write(struct.pack(endian_fmt+'i', nc_data['ny']))
        f.write(struct.pack(endian_fmt+'i', nc_data['iproj']))

        f.write(struct.pack(endian_fmt+'i', record_size))

        #record 3 , fmt = endian_fmt+"8sfffff"
        record_size = struct.calcsize(endian_fmt+"8sfffff")
        f.write(struct.pack(endian_fmt+'i', record_size))

        formatted_string = '{0:8s}'.format(nc_data['startloc'])
        f.write(struct.pack(endian_fmt+'8s', formatted_string.encode(string_encoding)))

        f.write(struct.pack(endian_fmt+'f', nc_data['startlat']))
        f.write(struct.pack(endian_fmt+'f', nc_data['startlon']))
        # For the Lat-Lon projection next field should be deltalat
        # //TODO: but for Gaussian projection this may have to be nlats instead
        # Check the user guide for intermidiate format
        f.write(struct.pack(endian_fmt+'f', nc_data['deltalat']))
        f.write(struct.pack(endian_fmt+'f', nc_data['deltalon']))
        f.write(struct.pack(endian_fmt+'f', nc_data['earth_radius']))

        f.write(struct.pack(endian_fmt+'i', record_size))
    else:
        raise Exception('WPS unnetcdf', 'unknown or not yet supported projection')

    #record 4
    f.write(struct.pack(endian_fmt+'i', 4))
    f.write(struct.pack(endian_fmt+'i', nc_data['is_wind_grid_rel']))
    f.write(struct.pack(endian_fmt+'i', 4))

    #record 5
    size = nc_data['nx'] * nc_data['ny']
    fmt = endian_fmt+"{}f".format(size)

    record_size = struct.calcsize(fmt)
    f.write(struct.pack(endian_fmt+'i', record_size))

    f.write(struct.pack(fmt, *nc_data['slab'].flatten('C')))

    f.write(struct.pack(endian_fmt+'i', record_size))

    # f.close()


def get_merra2_file_path(dataset_name, requested_date):
    MERRA2_STORAGE_PATH = '/home/osipovs/workspace/WRF/Data/merra2'
    MERRA2_STORAGE_PATH = '/project/k1090/osipovs/Data/NASA/MERRA2/'
    MERRA2_STORAGE_PATH = '/work/mm0062/b302074/Data/NASA/MERRA2/'
    name_prefix = 'MERRA2_400.'
    date_str = '.' + requested_date.strftime('%Y%m%d') + '.'
    if 'const_2d_asm' in dataset_name:
        name_prefix = 'MERRA2_101.'
        date_str = '.00000000.'
    elif 'const_2d_lnd' in dataset_name:
        name_prefix = 'MERRA2_100.'
        date_str = '.00000000.'
    nc_file_path = MERRA2_STORAGE_PATH + '/' + dataset_name + '/' + name_prefix + dataset_name + date_str + 'nc4'
    return nc_file_path


def get_emac_file_path(emac_folder, sim_label, dataset_name, requested_date, multifile_support, multifile_support_on_daily_output):
    '''
    The description of the EMAC sims https://gmd.copernicus.org/articles/9/1153/2016/


    RC2-oce-01 is:
     1. hindcast & projection 1950–1960–2100 with simulated SSTs/SICs
     2. with interactively coupled ocean model
     3. T42L47MA / GR30L40

    RC1SD-base-08 is:
     1. C1SD: hindcast 1979–1980–2013 with specified dynamics, ERA-Interim SSTs/SICs

    RC2 has prescribed antrho emissions: a combination of ACCMIP (Lamarque et al., 2010) and RCP 6.0 data


    pftp: https://www.dkrz.de/up/systems/hpss
    PATH: /hpss/arch/id0853/b302019

    :param dataset_name:
    :param requested_date:
    :return:
    '''
    # nc_file_path = '/work/id0853/b302019/{}/{}/{}_____{}_{}.nc'.format(sim_label, dataset_name, sim_label, requested_date.strftime('%Y%m'), dataset_name)  # 200204

    # I've downloaded sims again myself from hpss tape
    nc_file_name = '{}_____{}01_0000_{}.nc'.format(sim_label, requested_date.strftime('%Y%m'), dataset_name)
    nc_file_path = '/work/mm0062/b302074/Data/AirQuality/AQABA/CMIP6/EMAC_sims/RC2-oce-01/{}_{}/{}'.format(requested_date.strftime('%Y%m'), dataset_name, nc_file_name)

    nc_file_name = '{}{}_{}.nc'.format(sim_label, requested_date.strftime('%Y%m%d_0000'), dataset_name)  # daily pe file
    nc_file_path = '{}/{}'.format(emac_folder, nc_file_name)

    if multifile_support:
        nc_file_name = '{}{}_{}.nc'.format(sim_label, '*', dataset_name)  # requested_date.strftime('%Y%m%d_%H%M')
        nc_file_path = '{}/{}'.format(emac_folder, nc_file_name)

    if multifile_support_on_daily_output:
        nc_file_name = '{}{}_{}.nc'.format(sim_label, requested_date.strftime('%Y%m%d_*00'), dataset_name)  # daily per file
        nc_file_path = '{}/{}'.format(emac_folder, nc_file_name)

    return nc_file_path


_FIELD_MAP_EMAC_2_WRF = {
    # ECHAM5
    'TT': WrfMetgridMapItem('tpot', pp_impl=potential_to_regular_temperature),  # Temperature, K
    'HGT': WrfMetgridMapItem('geopot', pp_impl=convert_geopotential_to_height),  # Height (rho grid), m
    'PRESSURE': WrfMetgridMapItem('press'),  # Pressure, Pa
    'UU': WrfMetgridMapItem('um1', pp_impl=divide_by_cos_lat),
    'VV': WrfMetgridMapItem('vm1', pp_impl=divide_by_cos_lat),

    # In EMAC tsurf is skin temperature = sum of the land and sea temperature weighted by the land/ocean fractions
    # i.e. over ocean tsurf===sst, over land tsurf===land temperature
    # over the coast we will have a problem (inaccuracy)
    'SKINTEMP': WrfMetgridMapItem('tsurf'),  # skin temperature, K.

    # e5vdiff
    'T': WrfMetgridMapItem('temp2'),  # Temperature at 2 m, K

    # g3b
    'RH': WrfMetgridMapItem('relhum', pp_impl=convert_unit_ratio_to_percents),  # relative humidity, %
    # SURFACE variables
    'PSFC': WrfMetgridMapItem('aps'),  # surface pressure, Pa
    # TODO: 'PMSL': WrfMetgridMapItem('SLP', at_sea_level_pressure=True),  # mean sea level pressure, Pa, have to be pped in EMAC

    # UU and VV keys are taken by the 3d field, use U and V as a substitute for the 10m winds
    'U': WrfMetgridMapItem('u10'),  # U at 10 m
    'V': WrfMetgridMapItem('v10'),  # V at 10 m
    # TODO: rh_2m (relative) 'SPECHUMD': WrfMetgridMapItem('QV2M'),  # 2-meter_specific_humidity, kg kg**-1, Andrea did not find it

    'SEAICE': WrfMetgridMapItem('seaice'),  # ice covererd fraction of tile  # g3b
    # 'sst' : WrfMetgridMapItem('tsw'),  #  In EMAC tsurf===sst over ocean #TODO: alternative tsw

    'SOILHGT': WrfMetgridMapItem('geosp', pp_impl=convert_geopotential_to_height),  # surface geopotential height, m
    'LANDSEA': WrfMetgridMapItem('slm'),  # land is 1/sea is 0

    # Soil model
    # In this approach I've updated the METGRID.TBL and inserted custom depths
    # belowsf = 0.03, 0.19, 0.78, 2.68, 6.98 ;
    'ST000003': WrfMetgridMapItem('tsoil', pp_impl=split_soil_temperature),  # soil temperatures in layers 1-5, K
    'ST003019': WrfMetgridMapItem('tsoil', pp_impl=split_soil_temperature),
    'ST019078': WrfMetgridMapItem('tsoil', pp_impl=split_soil_temperature),
    'ST078268': WrfMetgridMapItem('tsoil', pp_impl=split_soil_temperature),
    'ST268698': WrfMetgridMapItem('tsoil', pp_impl=split_soil_temperature),

    # WRF treats soil temperature and humidity on the same vertical grid (see SOIL_LAYERS in the METGRID.TBL)
    # TODO: EMAC has simple bucket model, I need realistic conversation of ws and wsmx variables
    'SM000003': WrfMetgridMapItem('ws', pp_impl=split_soil_moisture),  # surface soil moisture, m**3 / m**3
    'SM003019': WrfMetgridMapItem('ws', pp_impl=split_soil_moisture),
    'SM019078': WrfMetgridMapItem('ws', pp_impl=split_soil_moisture),
    'SM078268': WrfMetgridMapItem('ws', pp_impl=split_soil_moisture),
    'SM268698': WrfMetgridMapItem('ws', pp_impl=split_soil_moisture),

    'SNOWH': WrfMetgridMapItem('sn'),  # Snow Depth, m
    #'SNOW': WrfMetgridMapItem('SNOMAS'),  # Water equivalent snow depth, kg m**-2

    # auxiliary translation of the variables and dimensions
    # they are axillary and are not exported to the intermediate format
    # I may need to introduce vars and dims separately
    'latitude': 'lat',
    'longitude': 'lon',
    # 'time' : 'time',
    'level': 'lev',
    }

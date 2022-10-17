import numpy as np


def derive_merra2_pressure_stag_profile(nc):
    # Build the entire 3d field first

    # see this doc for details on Vertical Structure https://gmao.gsfc.nasa.gov/pubs/docs/Bosilovich785.pdf
    layer_pressure_thickness = nc.variables['DELP']  # 3d

    # summation has to start from the top, p_top is fixed to 1 Pa
    pressure_stag_no_toa = 1 + np.cumsum(layer_pressure_thickness, axis=0)
    # new shape
    stag_shape = (pressure_stag_no_toa.shape[0]+1,) + pressure_stag_no_toa.shape[1:]
    pressure_stag = np.empty(stag_shape)
    pressure_stag[0, :, :] = 1  # fixed p_top
    pressure_stag[1:, :, :] = pressure_stag_no_toa

    # get the pressure at the rho grid
    pressure_rho = (pressure_stag[1:] + pressure_stag[:-1])/2

    return pressure_stag

# import ESMF
import xarray as xe

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

"""
This module contains utils to work with the grids and points in space 
"""

from numpy import unravel_index


def find_closest_grid_point(lon_grid, lat_grid, target_lon, target_lat):
    """
    L2 norm
    :param lon_grid:
    :param lat_grid:
    :param target_lon:
    :param target_lat:
    :return:
    """
    distance = (lat_grid - target_lat) ** 2 + (lon_grid - target_lon) ** 2
    minIndex = distance.argmin()
    tuple_index = unravel_index(minIndex, distance.shape)
    # compute the distance from a given target point to a given grid point
    distance_target = distance[tuple_index]
    return tuple_index, lon_grid[tuple_index], lat_grid[tuple_index], distance_target


def cell_area(ds):  # https://gist.github.com/JiaweiZhuang/fc2f860133174523051260817e75a99d
    '''

    Make sure to pass DS not DA

    Get cell area of a grid.
    Assume unit sphere (radius is 1, total area is 4*pi)
    Parameters
    ----------
    ds : xarray DataSet or dictionary
        Contains variables ``lon``, ``lat``, ``lon_b``, ``lat_b``
        Note that boundary is required for computing cell area
    Returns
    -------
    area : 2D numpy array for cell area
    '''

    grid = xe.frontend.ds_to_ESMFgrid(ds, need_bounds=True)
    grid = grid[0]
    field = ESMF.Field(grid)
    field.get_area()  # compute area

    # F-ordering to C-ordering
    # copy the array to make sure it persists after ESMF object is freed
    area = field.data.T.copy()
    field.destroy()

    return area

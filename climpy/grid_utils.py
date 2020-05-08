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
import datetime as dt
import numpy as np
import pandas as pd
import time

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'


def datetime_to_timestamp(time_data):
    timestamps = [time.mktime(date.timetuple()) for date in time_data]
    return timestamps


def average_time(time_data):
    if time_data.size == 0:
        return time_data
    timestamps = datetime_to_timestamp(time_data)
    average_date = dt.datetime.fromtimestamp(np.mean(timestamps))
    return average_date


def process_time_range_impl(time, time_range_vo):
    time_ind = np.logical_and(time >= time_range_vo.startDate, time < time_range_vo.endDate)
    time_ind = np.where(time_ind)[0]
    new_time = time[time_ind]
    t_slice = slice(time_ind[0], time_ind[-1] + 1)

    # distance = np.abs(timeData - timeRange.startDate)
    # minIndex = distance.argmin()
    #
    # distance = np.abs(timeData - timeRange.endDate)
    # maxIndex = distance.argmin()
    #
    # # if (minIndex == maxIndex):
    # maxIndex += 1
    #
    # timeSlice = slice(minIndex, maxIndex)
    #
    # newTimeData = timeData[timeSlice]

    return t_slice, new_time
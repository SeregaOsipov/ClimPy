from datetime import datetime
import numpy as np
import datetime as dt


def compute_daily_climatology(var_data, time_data, climatology_time_range_vo=None):
    if climatology_time_range_vo is None:
        ind = np.ones(time_data.shape, dtype='bool')
    else:
        ind = np.logical_and(time_data >= climatology_time_range_vo.startDate, time_data < climatology_time_range_vo.endDate)

    time_data_slice = time_data[ind]
    var_data_slice = var_data[ind]
    day_of_the_year_data = np.array([dateItem.timetuple().tm_yday for dateItem in time_data_slice])

    clim_data = np.ma.empty((366,) + var_data.shape[1:])
    clim_data[:] = np.NAN

    for i in range(366):
        ind = day_of_the_year_data == i + 1
        clim_data[i] = np.nanmean(var_data_slice[ind], axis=0)

    # if climatology only has 365 days, interpolate day 366 between 365th and 0th dayes
    unique_days = np.unique(day_of_the_year_data)
    if not 366 in unique_days:
        print("compute_daily_climatology: interpolating 366 from 1 and 355")
        clim_data[365] = (clim_data[364]+clim_data[0])/2

    # compute anomaly
    anomaly_data = np.ma.empty((len(time_data),) + var_data.shape[1:])
    anomaly_data[:] = np.NaN
    for i in range(len(time_data)):
        day = time_data[i].timetuple().tm_yday
        anomaly_data[i] = var_data[i] - clim_data[day - 1]

    return anomaly_data, clim_data, None, None


def compute_weekly_climatology(var_data, time_data, climatology_time_range_vo=None, convolve_data=True):
    if climatology_time_range_vo is None:
        ind = np.ones(time_data.shape, dtype='bool')
    else:
        ind = np.logical_and(time_data >= climatology_time_range_vo.startDate, time_data < climatology_time_range_vo.endDate)

    time_data_slice = time_data[ind]
    var_data_slice = var_data[ind]
    week_index_data = np.array([dateItem.timetuple().tm_yday/7 for dateItem in time_data_slice])
    # 364 = 7*52, only 2 days left for that week, merge them to the previous 51
    week_index_data[week_index_data==52]=51

    # there are 51 week in the year
    clim_data = np.ma.empty((52,) + var_data.shape[1:])
    clim_data[:] = np.NAN

    for i in range(52):
        ind = week_index_data == i
        clim_data[i] = np.nanmean(var_data_slice[ind], axis=0)

    # convolve data weekly
    weekly_var_data, weekly_time_data = convolve_data_weekly(var_data, time_data)
    # compute anomaly
    anomaly_data = np.ma.empty((len(weekly_time_data),) + var_data.shape[1:])
    anomaly_data[:] = np.NaN
    for i in range(len(weekly_time_data)):
        current_week_index = weekly_time_data[i].timetuple().tm_yday/7
        if current_week_index == 52:
            current_week_index = 51
        anomaly_data[i] = weekly_var_data[i] - clim_data[current_week_index]

    return anomaly_data, clim_data, weekly_var_data, weekly_time_data


def compute_monthly_climatology(var_data, time_data, climatology_time_range_vo=None, convolve_data=True):
    if climatology_time_range_vo is None:
        ind = np.ones(time_data.shape, dtype='bool')
    else:
        ind = np.logical_and(time_data >= climatology_time_range_vo.startDate, time_data < climatology_time_range_vo.endDate)

    time_data_slice = time_data[ind]
    var_data_slice = var_data[ind]
    month_index_data = np.array([dateItem.month for dateItem in time_data_slice])

    clim_data = np.ma.empty((12,) + var_data.shape[1:])
    clim_data[:] = np.NAN

    for i in range(12):
        ind = month_index_data == i + 1
        clim_data[i] = np.nanmean(var_data_slice[ind], axis=0)

    # convolve data monthly
    monthly_var_data = var_data
    monthly_time_data = time_data
    if convolve_data:
        monthly_var_data, monthly_time_data = convolve_data_monthly(var_data, time_data)
    # compute anomaly
    anomaly_data = np.ma.empty((len(monthly_time_data),) + var_data.shape[1:])
    anomaly_data[:] = np.NaN
    for i in range(len(monthly_time_data)):
        current_month_index = monthly_time_data[i].month
        anomaly_data[i] = monthly_var_data[i] - clim_data[current_month_index - 1]

    return anomaly_data, clim_data, monthly_var_data, monthly_time_data


def compute_quarterly_climatology(var_data, time_data, climatology_time_range_vo=None, convolve_data=True):
    if climatology_time_range_vo is None:
        ind = np.ones(time_data.shape, dtype='bool')
    else:
        ind = np.logical_and(time_data >= climatology_time_range_vo.startDate, time_data < climatology_time_range_vo.endDate)

    time_data_slice = time_data[ind]
    var_data_slice = var_data[ind]
    quarter_index_data = np.array([ (dateItem.month-1)/3+1 for dateItem in time_data_slice])

    # there are 4 quarters in the year
    clim_data = np.ma.empty((4,) + var_data.shape[1:])
    clim_data[:] = np.NAN

    for i in range(4):
        ind = quarter_index_data == i+1
        clim_data[i] = np.nanmean(var_data_slice[ind], axis=0)

    # convolve data
    quarterly_var_data, quarterly_time_data = convolve_data_quarterly(var_data, time_data)
    # compute anomaly
    anomaly_data = np.ma.empty((len(quarterly_time_data),) + var_data.shape[1:])
    anomaly_data[:] = np.NaN
    for i in range(len(quarterly_time_data)):
        current_quarter_index = int((quarterly_time_data[i].month-1)/3)
        anomaly_data[i] = quarterly_var_data[i] - clim_data[current_quarter_index]

    return anomaly_data, clim_data, quarterly_var_data, quarterly_time_data


def compute_yearly_climatology(var_data, time_data, climatology_time_range_vo=None, convolve_data=True):
    if climatology_time_range_vo is None:
        ind = np.ones(time_data.shape, dtype='bool')
    else:
        ind = np.logical_and(time_data >= climatology_time_range_vo.startDate, time_data < climatology_time_range_vo.endDate)

    yearly_data_slice, yearly_time_data_slice = convolve_data_yearly(var_data[ind], time_data[ind])
    clim_data = np.nanmean(yearly_data_slice, axis=0)

    # convolve data yearly
    yearly_data = var_data
    yearly_time_data = time_data
    if convolve_data:
        yearly_data, yearly_time_data = convolve_data_yearly(var_data, time_data)

    anomaly_data = yearly_data - clim_data
    return anomaly_data, clim_data, yearly_data, yearly_time_data


def convolve_data_weekly(var_data, time_data):
    # here we assume that time is sorted and growing and data is daily
    n_weeks = var_data.shape[0]/7
    reshaped_var_data = np.reshape(var_data[0:n_weeks*7], (var_data.shape[0]/7, 7,)+var_data.shape[1:])
    weekly_var_data = np.mean(reshaped_var_data, axis=1)
    weekly_time_data = time_data[0:n_weeks*7][3:-1:7]

    return weekly_var_data, weekly_time_data


def convolve_data_monthly(var_data, time_data):
    year_index_data = np.array([dateItem.year for dateItem in time_data])
    month_index_data = np.array([dateItem.month for dateItem in time_data])

    unique_years = np.unique(year_index_data)

    monthly_data = np.ma.empty((len(unique_years)*12,) + var_data.shape[1:])
    monthly_data[:] = np.NAN

    monthly_time_data = np.empty((len(unique_years) * 12,), dtype=datetime)
    monthly_time_data[:] = np.NAN

    counter = 0
    for year_index in range(len(unique_years)):
        for current_month_index in range(12):
            ind = np.logical_and(year_index_data == unique_years[year_index], month_index_data == current_month_index+1)
            monthly_data[counter] = np.nanmean(var_data[ind], axis=0)
            monthly_time_data[counter] = dt.datetime(unique_years[year_index], current_month_index+1, 15)
            counter += 1

    return monthly_data, monthly_time_data


def convolve_data_quarterly(var_data, time_data):
    year_index_data = np.array([dateItem.year for dateItem in time_data])
    quarter_index_data = np.array([ (dateItem.month-1)/3+1 for dateItem in time_data])

    unique_years = np.unique(year_index_data)

    quarterly_data = np.ma.empty((len(unique_years)*4,) + var_data.shape[1:])
    quarterly_data[:] = np.NAN

    quarterly_time_data = np.empty((len(unique_years) * 4,), dtype=datetime)
    quarterly_time_data[:] = np.NAN

    counter = 0
    for year_index in range(len(unique_years)):
        for current_quarter_index in range(4):
            ind = np.logical_and(year_index_data == unique_years[year_index], quarter_index_data==current_quarter_index+1)
            quarterly_data[counter] = np.nanmean(var_data[ind], axis=0)
            quarterly_time_data[counter] = dt.datetime(unique_years[year_index], current_quarter_index*3+1, 1)
            counter += 1

    return quarterly_data, quarterly_time_data


def convolve_data_yearly(var_data, time_data):
    year_index_data = np.array([dateItem.year for dateItem in time_data])
    unique_years = np.unique(year_index_data)

    yearly_data = np.ma.empty((len(unique_years),) + var_data.shape[1:])
    yearly_data[:] = np.NAN

    yearly_time_data = np.empty((len(unique_years),), dtype=datetime)
    yearly_time_data[:] = np.NAN

    for year_index in range(len(unique_years)):
        ind = year_index_data == unique_years[year_index]
        yearly_data[year_index] = np.nanmean(var_data[ind], axis=0)
        yearly_time_data[year_index] = dt.datetime(unique_years[year_index], 6, 1)

    return yearly_data, yearly_time_data
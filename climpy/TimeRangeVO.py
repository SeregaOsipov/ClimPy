from numpy.core.numeric import Infinity

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'

# TODO: this class will be removed in the future


class TimeRangeVO:
    startDate = -Infinity
    endDate = Infinity

    def __init__(self, start_date, end_date):
        self.startDate = start_date
        self.endDate = end_date

    def __repr__(self):
        return "TimeRangeVO. start date is: " + str(self.startDate) + ', end date is : ' + str(self.endDate)

    def __str__(self):
        return "TimeRangeVO. start date is: " + str(self.startDate) + ', end date is : ' + str(self.endDate)



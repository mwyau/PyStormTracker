import math
import numpy as np
import numpy.ma as ma
from abc import ABCMeta, abstractmethod
from scipy.ndimage.filters import generic_filter, laplace
import Nio

class Grid(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_var(self):
        raise NotImplementedError

    @abstractmethod
    def get_time(self):
        raise NotImplementedError

    @abstractmethod
    def get_lat(self):
        raise NotImplementedError

    @abstractmethod
    def get_lon(self):
        raise NotImplementedError

    @abstractmethod
    def split(self, num):
        raise NotImplementedError

    @abstractmethod
    def detect(self):
        raise NotImplementedError

class Center(object):

    R = 6367.
    DEGTORAD = math.pi/180.

    def __init__(self, time, lat, lon, var):
        self.time =time
        self.lat = lat
        self.lon = lon
        self.var = var

    def __repr__(self):
        return str(self.var)

    def __str__(self):
        return "[time="+str(self.time)+", lat="+str(self.lat)+", lon="+ \
                str(self.lon)+ ", var="+str(self.var)+"]"

    def abs_dist(self, center):
        """Haversine formula for calculating the great circle distance"""

        if not isinstance(center, Center):
            raise TypeError, "must be compared with a Center object"

        dlat = center.lat - self.lat
        dlon = center.lon - self.lon

        return self.R*2*math.asin(math.sqrt(math.sin(dlat/2*self.DEGTORAD)**2 + \
                math.cos(self.lat*self.DEGTORAD) * math.cos(center.lat*self.DEGTORAD) * \
                math.sin(dlon/2*self.DEGTORAD)**2))

    def lat_dist(self, center):

        if not isinstance(center, Center):
            raise TypeError, "must be compared with a Center object"

        dlat = center.lat - self.lat

        return self.R*dlat*self.DEGTORAD

    def lon_dist(self, center):

        if not isinstance(center, Center):
            raise TypeError, "must be compared with a Center object"

        avglat = (self.lat + center.lat)/2
        dlon = center.lon - self.lon

        return self.R*dlon*self.DEGTORAD*math.cos(avglat*self.DEGTORAD)

class RectGrid(Grid):

    def __init__(self, pathname, varname, trange=None):

        self.pathname = pathname
        self.varname = varname
        self.trange = trange

        self._open_file = False
        self._var = None
        self._time = None
        self._lat = None
        self._lon = None

    def _init(self):

        if self._open_file is False:
            self._open_file = True
            self.f = Nio.open_file(self.pathname)

            # Dimension of var is time, lat, lon
            self._var = self.f.variables[self.varname]
            self._time = self.f.variables['time']
            self._lat = self.f.variables['latitude']
            self._lon = self.f.variables['longitude']
            self.time = None
            self.lat = None
            self.lon = None

    def get_var(self, chart=None):

        if self.trange is not None:
            if self.trange[0] >= self.trange[1]:
                return None

        if chart is not None:
            if type(chart) is tuple:
                if len(chart) != 2 or type(chart[0]) is not int or \
                        type(chart[1]) is not int:
                    raise TypeError, "chart must be a tuple of two integers"
            elif type(chart) is not int:
                raise TypeError, "chart must be an integer or tuple"

            if self.trange is not None:
                if type(chart) is int:
                    if chart < 0 or chart >= self.trange[1]-self.trange[0]:
                        raise IndexError, "chart is out of bound of trange"
                if type(chart) is tuple:
                    if chart[0] == chart[1]:
                        return None
                    if chart[0] > chart[1]:
                        raise IndexError, "chart[1] must be larger than chart[0]"
                    if chart[0] <0 or chart[0] > self.trange[1]-self.trange[0]:
                        raise IndexError, "chart[0] is out of bound of trange"
                    if chart[1] <0 or chart[1] > self.trange[1]-self.trange[0]:
                        raise IndexError, "chart[1] is out of bound of trange"

        self._init()

        if type(chart) is int:
            if self.trange is None:
                return self._var[chart,:,:]
            else:
                return self._var[self.trange[0]+chart,:,:]
        elif type(chart) is tuple:
            if self.trange is None:
                return self._var[chart[0]:chart[1],:,:]
            else:
                return self._var[self.trange[0]+chart[0]:self.trange[0]+chart[1],:,:]
        else:
            return self._var[:]

    def get_time(self):

        if self.trange is not None:
            if self.trange[0] >= self.trange[1]:
                return None

        self._init()
        if self.time is None:
            if self.trange is None:
                self.time = self._time[:]
            else:
                self.time = self._time[self.trange[0]:self.trange[1]]
        return self.time

    def get_lat(self):

        self._init()
        if self.lat is None:
            self.lat = self._lat[:]
        return self.lat

    def get_lon(self):

        self._init()
        if self.lon is None:
            self.lon = self._lon[:]
        return self.lon

    def split(self, num):

        if type(num) is not int:
            raise TypeError, "number to split must be an integer"

        if self._open_file is False:

            if self.trange is not None:
                time_len = self.trange[1]-self.trange[0]
                tstart = self.trange[0]
            else:
                f = Nio.open_file(self.pathname)
                time_len = f.dimensions['time']
                f.close()
                tstart = 0

            chunk_size = time_len//num
            remainder = time_len%num

            tranges = [(tstart+i*chunk_size+remainder*i//num, \
                    tstart+(i+1)*chunk_size+remainder*(i+1)//num) \
                    for i in range(num)]

            return [RectGrid(self.pathname, self.varname, trange=it) \
                    for it in tranges]

        else:
            raise RuntimeError, "RectGrid must not be initialized before running split()"

    def _local_extrema_func(self, buffer, size, threshold, minmaxmode):

        half_size = size//2

        search_window = buffer.reshape((size, size))
        origin = (half_size, half_size)

        if threshold == 0.:
            if minmaxmode == 'min':
                return search_window[origin] == search_window.min()
            elif minmaxmode == 'max':
                return search_window[origin] == search_window.max()
        elif search_window[origin] == search_window.min():
            if minmaxmode == 'min':
                # At least 8 of values in buffer should be larger than threshold
                return sorted(buffer)[8] - search_window[origin] > threshold
        elif search_window[origin] == search_window.max():
            if minmaxmode == 'max':
                return sorted(buffer)[0] - search_window[origin] < -1*threshold
        return False

    def _local_extrema_filter(self, input, size, threshold=0., minmaxmode='min'):

        if size%2 != 1:
            raise ValueError, "size must be an odd number"

        half_size = size//2

        output = generic_filter(input, self._local_extrema_func, size=size, \
                mode='wrap', extra_keywords={'size': size, 'threshold': threshold, 'minmaxmode': minmaxmode})

        # Mask the extreme latitudes
        output[:half_size,:] = 0.
        output[-half_size:,:] = 0.

        return output

    def _local_max_laplace(self, buffer, size):
        origin = (size*size)//2
        return buffer[origin] and buffer[origin] == buffer.max()

    def _remove_dup_laplace(self, data, mask, size=5):
        laplacian = np.multiply(laplace(data, mode='wrap'), mask)

        return generic_filter(laplacian, self._local_max_laplace, size=size, mode='wrap',
                extra_keywords={'size': size})

    def detect(self, size=5, threshold=0., chart_buffer=400, minmaxmode='min'):

        """Returns a list of list of Center's"""

        if self.trange is not None and self.trange[0]>=self.trange[1]:
            return []

        time = self.get_time()
        lat = self.get_lat()
        lon = self.get_lon()

        centers = []

        for it, t in enumerate(time):

            print "Working on it:", str(it)

            ibuffer = it%chart_buffer
            if ibuffer == 0:
                var = self.get_var(chart=(it,min(it+chart_buffer,len(time))))
            chart = var[ibuffer,:,:]

            extrema = self._local_extrema_filter(chart, size, threshold=threshold, minmaxmode=minmaxmode)
            extrema = self._remove_dup_laplace(chart, extrema, size=5)

            center_list = [Center(t, lat[i], lon[j], chart[i,j]) \
                    for i, j in np.transpose(extrema.nonzero())]
            centers.append(center_list)

        return centers

if __name__ == "__main__":

    grid = RectGrid(pathname="../slp.2012.nc", varname="slp")
    grids = grid.split(128)

    for g in grids:
        print(g.trange)

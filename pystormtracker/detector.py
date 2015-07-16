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

    def __str__(self):
        return str(self.var)

    __repr__ = __str__

    @classmethod
    def abs_dist(cls, center1, center2):
        """Haversine formula for calculating the great circle distance"""

        dlat = center2.lat - center1.lat
        dlon = center2.lon - center1.lon

        return cls.R*2*math.asin(math.sqrt(math.sin(dlat/2*cls.DEGTORAD)**2 + \
                math.cos(center1.lat*cls.DEGTORAD) * math.cos(center2.lat*cls.DEGTORAD) * \
                math.sin(dlon/2*cls.DEGTORAD)**2))

    @classmethod
    def lat_dist(cls, center1, center2):

        dlat = center2.lat - center1.lat

        return cls.R*dlat*cls.DEGTORAD

    @classmethod
    def lon_dist(cls, center1, center2):

        avglat = (center1.lat + center2.lat)/2
        dlon = center2.lon - center1.lon

        return cls.R*dlon*cls.DEGTORAD*math.cos(avglat*cls.DEGTORAD)

class RectGrid(Grid):

    def __init__(self, pathname, varname, trange=None, open_file=True):

        self.pathname = pathname
        self.varname = varname
        self.trange = trange

        if open_file:
            self._open_file()
        else:
            self._var = None
            self._time = None
            self._lat = None
            self._lon = None

    def _open_file(self):

        f = Nio.open_file(self.pathname)

        # Dimension of var is time, lat, lon
        self._var = f.variables[self.varname]
        self._time = f.variables['time']
        self._lat = f.variables['lat']
        self._lon = f.variables['lon']
        if self.trange:
            self.time = self._time[self.trange[0]:self.trange[1]]
        else:
            self.time = self._time[:]
        self.lat = self._lat[:]
        self.lon = self._lon[:]

    def get_var(self):

        if self._var is None:
            self._open_file()
        if self.trange:
            return self._var[self.trange[0]:self.trange[1],:,:]
        else:
            return self._var[:]

    def get_time(self):
        if self._time is None:
            self._open_file()
        else:
            return self.time

    def get_lat(self):
        if self._lat is None:
            self._open_file()
        return self.lat

    def get_lon(self):
        if self._lon is None:
            self._open_file()
        return self.lon

    def _local_minima_func(self, buffer, size, threshold):

        half_size = size//2

        search_window = buffer.reshape((size, size))
        origin = (half_size, half_size)

        if threshold == 0.:
            return search_window[origin] == search_window.min()
        elif search_window[origin] == search_window.min():
            # # At least 1/2 of values in buffer should be larger than threshold
            # At least 8 of values in buffer should be larger than threshold
            return sorted(buffer)[8] - search_window[origin] > threshold
        else:
            return False

    def _local_minima_filter(self, input, size, threshold=0.):

        assert size%2 == 1, "size must be an odd number"
        half_size = size//2

        output = generic_filter(input, self._local_minima_func, size=size, \
                mode='wrap', extra_keywords={'size': size, 'threshold': threshold})

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

    def detect(self, size=5, threshold=0.):

        time = self.get_time()
        lat = self.get_lat()
        lon = self.get_lon()

        centers = []

        for it, t in enumerate(time):

            chart = self.get_var()[it,:,:]
            minima = self._local_minima_filter(chart, size, threshold=threshold)
            minima = self._remove_dup_laplace(chart, minima, size=5)

            for i, j in np.transpose(minima.nonzero()):
                c = Center(t, lat[i], lon[j], chart[i,j])
                centers.append(c)

        return centers

    def split(self, num):
        pass

if __name__ == "__main__":

    grid = RectGrid(pathname="../slp.2012.nc", varname="slp", trange=(0,1))
    centers = grid.detect()

    print(centers)

import math
import numpy as np
import numpy.ma as ma
from scipy.ndimage.filters import generic_filter, laplace
import Nio

class Center(object):

    R = 6367.
    DEGTORAD = math.pi/180.

    def __init__(self, time, lat, lon, var):
        self.time =time
        self.lat = lat
        self.lon = lon
        self.var = var

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

def get_var_handle(filename, varname="slp"):

    f = Nio.open_file(filename)

    # Dimension of var is lat, lon)
    var = f.variables[varname]
    time = f.variables['time']
    lat = f.variables['lat']
    lon = f.variables['lon']

    return var, time, lat, lon

def local_minima_filter_latlon(input, size, threshold=0.):

    # Need to add error check: assert(size is an odd number)

    half_size = size//2

    def local_min_latlon(buffer, size, threshold=threshold):
        search_window = buffer.reshape((size, size))
        origin = (half_size, half_size)
        if threshold == 0.:
            return search_window[origin] == search_window.min()
        elif search_window[origin] == search_window.min():
            # At least 1/2 of values in buffer should be larger than threshold
            return sorted(buffer)[size//2] - search_window[origin] > threshold
        else:
            return False

    input_shape = input.shape
    padded_shape = (input_shape[0]+half_size*2,input_shape[1])

    padded_input = ma.empty(padded_shape)

    padded_input[:half_size,] = np.nan
    padded_input[half_size:-half_size,] = input
    padded_input[-half_size:,] = np.nan

    padded_output = generic_filter(padded_input, local_min_latlon, size=size,
            mode='wrap', extra_keywords={'size': size})

    return padded_output[half_size:-half_size,]

def remove_dup_laplace_latlon(data, mask, size=5):
    laplacian = np.multiply(laplace(data, mode='wrap'), mask)

    def local_max_laplace(buffer, size):
        origin = (size*size)//2
        return buffer[origin] and buffer[origin] == buffer.max()

    return generic_filter(laplacian, local_max_laplace, size=size, mode='wrap',
            extra_keywords={'size': size})

def detect_center_latlon(data, size=5, threshold=0.):

    minima = local_minima_filter_latlon(data, size, threshold=threshold)
    minima = remove_dup_laplace_latlon(data, minima, size=5)

    return minima

if __name__ == "__main__":
    var, time, lat, lon = get_var_handle(filename="../slp.2012.nc")
    centers = detect_center_latlon(var[0,:,:],threshold=200.)
    center_indices = centers.nonzero()
    print(np.transpose(center_indices))
    print(time[:])
    print(lat[:])
    print(lon[:])

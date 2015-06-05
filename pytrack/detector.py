import numpy as np
import numpy.ma as ma
from scipy.ndimage.filters import generic_filter, laplace
import Nio

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

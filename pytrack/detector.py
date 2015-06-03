import numpy as np
import numpy.ma as ma
from scipy.ndimage.filters import generic_filter
import Nio

def get_var_handle(filename, varname="slp"):

    f = Nio.open_file(filename)

    # Dimension of var is lat, lon)
    var = f.variables[varname]
    time = f.variables['time']
    lat = f.variables['lat']
    lon = f.variables['lon']

    return var, time, lat, lon

def local_minima_filter_latlon(input, size):

    # Need to add error check: assert(size is an odd number)

    half_size = size//2

    def local_minima_latlon(buffer, size):
        search_window = buffer.reshape((size, size))
        origin = (half_size, half_size)
        return search_window[origin] == search_window.min()

    input_shape = input.shape
    padded_shape = (input_shape[0]+half_size*2,input_shape[1])

    padded_input = ma.empty(padded_shape)

    padded_input[:half_size,] = np.nan
    padded_input[half_size:-half_size,] = input
    padded_input[-half_size:,] = np.nan

    padded_output = generic_filter(padded_input, local_minima_latlon, size = size,
            mode = 'wrap', extra_keywords = {'size': size})

    return padded_output[half_size:-half_size,]

def detect_center_latlon(data, size=5):

    minima = local_minima_filter_latlon(data, size)

    return minima

if __name__ == "__main__":
    data, time, lat, lon = get_var_handle(filename="../slp.2012.nc")
    centers = detect_center_latlon(data[0,:,:])
    center_indices = centers.nonzero()
    print(np.transpose(center_indices))
    print(time[:])
    print(lat[:])
    print(lon[:])

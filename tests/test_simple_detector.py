import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from pystormtracker.simple.detector import SimpleDetector

@patch('netCDF4.Dataset')
def test_simple_detector_init(mock_dataset):
    mock_f = MagicMock()
    mock_dataset.return_value = mock_f
    
    mock_var = MagicMock()
    mock_f.variables = {
        'slp': mock_var,
        'time': MagicMock(),
        'lat': MagicMock(),
        'lon': MagicMock()
    }
    
    detector = SimpleDetector(pathname="test.nc", varname="slp")
    detector._init()
    
    mock_dataset.assert_called_once_with("test.nc", "r")
    mock_var.set_auto_maskandscale.assert_called_once_with(False)

@patch('netCDF4.Dataset')
def test_simple_detector_detect_mock(mock_dataset):
    mock_f = MagicMock()
    mock_dataset.return_value = mock_f
    
    # Create mock data: a 5x5 grid with a single minimum in the center
    # Use 7x7 to allow for filter size 5 and masking of extremes
    data = np.ones((1, 7, 7)) * 1000
    data[0, 3, 3] = 950 # Minimum at index 3,3
    
    mock_var = MagicMock()
    mock_var.__getitem__.side_effect = lambda x: data[x]
    mock_var.set_auto_maskandscale = MagicMock()
    
    mock_time = np.array([0.0])
    mock_lat = np.array([0, 1, 2, 3, 4, 5, 6])
    mock_lon = np.array([0, 1, 2, 3, 4, 5, 6])
    
    mock_f.variables = {
        'slp': mock_var,
        'time': mock_time,
        'lat': mock_lat,
        'lon': mock_lon
    }
    
    detector = SimpleDetector(pathname="test.nc", varname="slp")
    # Manually set attributes to skip _init if needed or let it run
    
    centers = detector.detect(size=5, threshold=0.0)
    
    assert len(centers) == 1
    assert len(centers[0]) == 1
    center = centers[0][0]
    assert center.lat == 3.0
    assert center.lon == 3.0
    assert center.var == 950.0

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from pystormtracker.hodges.tracker import HodgesTracker
from pystormtracker.io.data_loader import DataLoader
from pystormtracker.metrics.compare import TrackComparisonConfig, compare_tracks
from pystormtracker.preprocessing.regrid import SpectralRegridder
from pystormtracker.preprocessing.spectral import SHTFilter
from tests.utils import (
    DECEMBER_2025_END,
    DECEMBER_2025_START,
    fetch_era5_msl,
    fetch_era5_vo850,
)

pytestmark = [pytest.mark.integration, pytest.mark.data]


@pytest.fixture(scope="module")
def n320_msl_path() -> str:
    """Download N320 mean sea level pressure data once per module."""
    pytest.importorskip("cfgrib")
    return fetch_era5_msl(resolution="n320", format="grib")


@pytest.fixture(scope="module")
def n320_vo_path() -> str:
    """Download N320 relative vorticity data once per module."""
    pytest.importorskip("cfgrib")
    return fetch_era5_vo850(resolution="n320", format="grib")


@pytest.fixture(scope="module")
def regular_vo_path() -> str:
    """Download 0.25-degree relative-vorticity data once per module."""
    return fetch_era5_vo850(resolution="0.25x0.25")


@pytest.mark.integration
def test_reduced_gaussian_loader(n320_msl_path: str) -> None:
    loader = DataLoader(n320_msl_path)
    loader.ensure_open()

    assert loader.is_reduced_gaussian("msl")
    pl = loader.get_reduced_grid_pl("msl")
    assert pl is not None
    assert len(pl) == 640
    assert np.sum(pl) == 542080


@pytest.mark.integration
def test_reduced_gaussian_vo_loader(n320_vo_path: str) -> None:
    """Verify the N320 relative-vorticity test dataset exposes reduced-grid metadata."""
    loader = DataLoader(n320_vo_path)
    loader.ensure_open()

    assert loader.is_reduced_gaussian("vo")
    pl = loader.get_reduced_grid_pl("vo")
    assert pl is not None
    assert len(pl) == 640
    assert np.sum(pl) == 542080


@pytest.mark.integration
def test_reduced_gaussian_filter_to_cc(n320_msl_path: str) -> None:
    ds = xr.open_dataset(n320_msl_path, engine="cfgrib")
    # Process only first 2 time steps to keep it fast
    data = ds.msl.isel(time=slice(0, 2))

    # Filter and regrid to a 1 degree Lat-Lon grid (181x360)
    filtered = SHTFilter(
        lmin=5, lmax=42, out_geometry="CC", out_ntheta=181, out_nphi=360
    ).filter(data)

    assert filtered.dims == ("time", "latitude", "longitude")
    assert filtered.shape == (2, 181, 360)
    assert not np.isnan(filtered.values).any()


@pytest.mark.integration
def test_reduced_gaussian_filter_to_gl(n320_msl_path: str) -> None:
    ds = xr.open_dataset(n320_msl_path, engine="cfgrib")
    data = ds.msl.isel(time=0)

    # Filter and regrid to a regular N80 Gaussian grid (160x320)
    filtered = SHTFilter(
        lmin=0, lmax=80, out_geometry="GL", out_ntheta=160, out_nphi=320
    ).filter(data)

    assert filtered.dims == ("latitude", "longitude")
    assert filtered.shape == (160, 320)


@pytest.mark.integration
def test_reduced_gaussian_regridder(n320_msl_path: str) -> None:
    ds = xr.open_dataset(n320_msl_path, engine="cfgrib")
    data = ds.msl.isel(time=0)

    regridder = SpectralRegridder(lmax=80)

    # Regrid to HEALPix nside=32
    hp_data = regridder.to_healpix(
        data,
        nside=32,
    )

    assert hp_data.dims == ("cell",)
    assert hp_data.shape == (12 * 32**2,)


@pytest.mark.integration
def test_reduced_gaussian_tracking_flow(n320_msl_path: str) -> None:
    tracker = HodgesTracker(
        min_track_points=3,
        min_object_grid_points=1,
    )

    ds = xr.open_dataset(n320_msl_path, engine="cfgrib")
    # Take a small slice for speed
    data = ds.msl.isel(time=slice(0, 10))

    # 1. Manually filter and regrid to CC
    data_filtered = SHTFilter(
        lmin=5, lmax=42, out_geometry="CC", out_ntheta=181, out_nphi=360
    ).filter(data)

    # 2. Track on the regridded data
    tracks = tracker.track(
        data=data_filtered,
        variable="msl",
        detection_mode="min",
        object_threshold=101000.0,  # Pa
    )

    assert len(tracks) > 0
    assert any(len(t) >= 3 for t in tracks)


@pytest.mark.integration
@pytest.mark.slow
def test_hodges_vorticity_tracks_agree_between_n320_and_regular_grid(
    n320_vo_path: str, regular_vo_path: str
) -> None:
    """Compare Hodges trajectories on paired ERA5 vorticity fields."""
    pytest.importorskip("cfgrib")
    n320 = xr.open_dataset(n320_vo_path, engine="cfgrib").vo
    regular = xr.open_dataset(regular_vo_path, engine="h5netcdf").vo.squeeze(drop=True)
    n320_time_dim = DataLoader(n320).get_coords()[0]
    regular_time_dim = DataLoader(regular).get_coords()[0]
    common_times = np.intersect1d(
        n320[n320_time_dim].values, regular[regular_time_dim].values
    )
    assert common_times.size > 0
    selected_times = common_times[
        (common_times >= np.datetime64(DECEMBER_2025_START))
        & (common_times <= np.datetime64(DECEMBER_2025_END))
    ]
    assert selected_times.size == 124
    assert selected_times[0] == np.datetime64(DECEMBER_2025_START)
    assert selected_times[-1] == np.datetime64(DECEMBER_2025_END)

    n320_common = n320.sel({n320_time_dim: selected_times})
    regular_common = regular.sel({regular_time_dim: selected_times})
    sht_filter = SHTFilter(
        lmin=5,
        lmax=42,
        out_geometry="CC",
        out_ntheta=181,
        out_nphi=360,
    )
    n320_filtered = sht_filter.filter(n320_common)
    regular_filtered = sht_filter.filter(regular_common)
    np.testing.assert_array_equal(
        n320_filtered[n320_time_dim].values,
        regular_filtered[regular_time_dim].values,
    )

    tracker = HodgesTracker(min_track_points=3)
    n320_tracks = tracker.track(
        data=n320_filtered,
        variable="vo",
        detection_mode="max",
        object_threshold=1.0e-4,
    )
    regular_tracks = tracker.track(
        data=regular_filtered,
        variable="vo",
        detection_mode="max",
        object_threshold=1.0e-4,
    )
    assert len(n320_tracks) > 0
    assert len(regular_tracks) > 0

    comparison = compare_tracks(
        regular_tracks,
        n320_tracks,
        config=TrackComparisonConfig(variable="vo"),
    )
    mean_separations_km = np.asarray(
        [match.mean_separation_km for match in comparison.matches], dtype=np.float64
    )
    mean_eligible_candidates = float(
        np.mean([match.eligible_candidate_count for match in comparison.matches])
    )
    duration_differences_hours = np.asarray(
        [
            match.candidate.duration_hours - match.reference.duration_hours
            for match in comparison.matches
        ],
        dtype=np.float64,
    )
    peak_intensity_differences = np.asarray(
        [
            match.candidate.peak_intensity - match.reference.peak_intensity
            for match in comparison.matches
            if match.candidate.peak_intensity is not None
            and match.reference.peak_intensity is not None
        ],
        dtype=np.float64,
    )
    print(
        "Hodges 0.25-degree reference vs N320 candidate: "
        f"{comparison.match_count}/{len(regular_tracks)} reference "
        f"({comparison.reference_coverage:.3f}), "
        f"{len(n320_tracks) - len(comparison.unmatched_candidate_ids)}/"
        f"{len(n320_tracks)} candidate selected "
        f"({comparison.candidate_coverage:.3f}), "
        f"mean eligible candidates={mean_eligible_candidates:.3f}; "
        f"mean separation={np.mean(mean_separations_km):.3f} km, "
        f"p95={np.percentile(mean_separations_km, 95):.3f} km; "
        f"median duration difference={np.median(duration_differences_hours):.3f} h, "
        "median peak-vorticity difference="
        f"{np.median(peak_intensity_differences):.3e} s^-1"
    )
    assert comparison.reference_coverage >= 0.9
    for match in comparison.matches:
        assert match.overlap_fraction >= comparison.config.min_overlap_fraction
        assert match.mean_separation_deg <= comparison.config.max_mean_separation_deg
        assert np.isfinite(
            [
                match.reference.duration_hours,
                match.reference.path_length_km,
                match.reference.mean_speed_kmh,
                match.candidate.duration_hours,
                match.candidate.path_length_km,
                match.candidate.mean_speed_kmh,
                match.mean_separation_km,
            ]
        ).all()
        assert match.intensity_difference is not None
        assert np.isfinite(
            [
                match.intensity_difference.bias,
                match.intensity_difference.mae,
                match.intensity_difference.rmse,
            ]
        ).all()

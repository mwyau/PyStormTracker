from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ..models import constants as model_constants
from ..models.tracker import RawDetectionStep, Tracker
from ..models.tracks import Tracks
from ..preprocessing.taper import TaperFilter
from . import constants
from .detector import HodgesDetector
from .linker import HodgesLinker

if TYPE_CHECKING:
    from ..models.geo import MapExtent


class HodgesTracker(Tracker):
    """
    A tracker implementing the Hodges (TRACK) algorithm with adaptive constraints.
    """

    def __init__(
        self,
        w1: float = constants.W1_DEFAULT,
        w2: float = constants.W2_DEFAULT,
        dmax: float = constants.DMAX_DEFAULT,
        phimax: float = constants.PHIMAX_DEFAULT,
        n_iterations: int = constants.ITERATIONS_DEFAULT,
        min_lifetime: int = constants.LIFETIME_DEFAULT,
        max_missing: int = constants.MISSING_DEFAULT,
        zones: NDArray[np.float64] | None = None,
        adapt_params: NDArray[np.float64] | None = None,
        use_standard_constraints: bool = True,
    ) -> None:
        """
        Initialize the Hodges Tracker.

        Args:
            w1: Weight for direction in cost function.
            w2: Weight for speed in cost function.
            dmax: Default maximum displacement in degrees.
            phimax: Penalty for phantom points (static cost).
            n_iterations: Number of MGE iterations (forward + backward).
            min_lifetime: Minimum number of steps for a valid track.
            max_missing: Maximum consecutive missing frames allowed.
            zones: Regional dmax zones [lon_min, lon_max, lat_min, lat_max, dmax].
            adapt_params: Adaptive smoothness parameters (2x4 array).
            use_standard_constraints: If True, use legacy standard zones/adaptive
                values if None provided.
        """
        self.w1 = w1
        self.w2 = w2
        self.dmax = dmax
        self.phimax = phimax
        self.n_iterations = n_iterations
        self.min_lifetime = min_lifetime
        self.max_missing = max_missing

        if zones is None:
            if use_standard_constraints:
                self.zones = constants.TRACK_ZONES
            else:
                self.zones = np.zeros((0, 5), dtype=np.float64)
        else:
            self.zones = zones

        if adapt_params is None:
            if self.phimax > 0:
                self.adapt_params = constants.ADAPT_PARAMS
            else:
                self.adapt_params = np.zeros((2, 0), dtype=np.float64)
        else:
            self.adapt_params = adapt_params

    def preprocess_standard_track(
        self,
        data: xr.DataArray,
        lmin: int = constants.LMIN_DEFAULT,
        lmax: int = constants.LMAX_DEFAULT,
        taper_points: int = constants.TAPER_DEFAULT,
        map_proj: Literal["global", "nh_stereo", "sh_stereo", "healpix"] = "global",
        resolution: float = 100.0,
        extent: MapExtent | None = None,
        filter_type: Literal["sht", "dct", "auto"] = "auto",
    ) -> xr.DataArray:
        """
        Applies standard TRACK preprocessing: Tapering -> SHT or DCT Filter.
        Optionally regrids to a Polar Stereographic or HEALPix projection.
        """
        from ..io.data_loader import DataLoader
        from ..preprocessing.spectral import DCTFilter, SHTFilter

        loader = DataLoader(data.dataset if hasattr(data, "dataset") else data)
        _time_dim, _lat_dim, _lon_dim = loader.get_coords()

        if filter_type == "auto":
            filter_type = "sht" if loader.is_global_longitude() else "dct"

        # Ensure data is loaded into memory for spectral filtering
        if data.chunks:
            data = data.compute()

        # 1. Tapering
        if taper_points > 0:
            taper = TaperFilter(n_points=taper_points)
            data = cast(xr.DataArray, taper.filter(data))

        # 2. Regridding and Filtering
        if map_proj in ("nh_stereo", "sh_stereo", "healpix"):
            from ..preprocessing.regrid import SpectralRegridder

            regridder = SpectralRegridder(lmax=lmax)
            is_lat_reversed = loader.is_lat_reversed()

            time_dim = next(
                (c for c in DataLoader.VAR_MAPPING["time"] if c in data.dims), "time"
            )

            out_frames = []
            for i in range(len(data[time_dim])):
                frame = data.isel({time_dim: i}).squeeze()
                if map_proj == "healpix":
                    nside = int(
                        np.sqrt(12 * (lmax + 1) ** 2 / 12)
                    )  # Rough heuristic, can be customized
                    nside = 2 ** int(
                        np.round(np.log2(max(1, nside)))
                    )  # Round to power of 2
                    # Note: filter_lmin is not directly in to_healpix currently,
                    # but we can filter first
                    if lmin > 0:
                        f_obj = SHTFilter(lmin=lmin, lmax=lmax)
                        frame = f_obj.filter(frame)
                    out_frame = regridder.to_healpix(
                        frame, nside=nside, lat_reverse=is_lat_reversed
                    )
                else:
                    hemi: Literal["nh", "sh"] = (
                        "nh" if map_proj == "nh_stereo" else "sh"
                    )

                    out_frame = regridder.to_polar_stereo(
                        frame,
                        hemisphere=hemi,
                        filter_lmin=lmin if lmin > 0 else None,
                        lmax=lmax,
                        lat_reverse=is_lat_reversed,
                        resolution=resolution,
                        extent=extent
                        if extent is not None
                        else (-13000.0, 13000.0, -13000.0, 13000.0),
                    )
                out_frames.append(out_frame)
            # Concatenate back
            data = xr.concat(out_frames, dim=data[time_dim])
            data.attrs["map_proj"] = map_proj
        else:
            # Global or regional grid filtering
            f_cls = SHTFilter if filter_type == "sht" else DCTFilter
            spectral_filter = f_cls(lmin=lmin, lmax=lmax)
            data = spectral_filter.filter(data)

        return data

    def track(
        self,
        infile: str | Path | xr.DataArray | xr.Dataset,
        varname: str,
        start_time: str | np.datetime64 | None = None,
        end_time: str | np.datetime64 | None = None,
        mode: Literal["min", "max"] = "min",
        map_proj: Literal["global", "nh_stereo", "sh_stereo", "healpix"] = "global",
        resolution: float = 100.0,
        extent: MapExtent | None = None,
        backend: Literal["serial", "mpi", "dask"] = "serial",
        n_workers: int | None = None,
        max_chunk_size: int | None = None,
        threshold: float | None = None,
        engine: str | None = None,
        overlap: int = model_constants.OVERLAP_DEFAULT,
        min_points: int = constants.MIN_POINTS_DEFAULT,
        filter: bool = True,
        lmin: int = constants.LMIN_DEFAULT,
        lmax: int = constants.LMAX_DEFAULT,
        taper_points: int = constants.TAPER_DEFAULT,
        subgrid_refine: bool = True,
        **kwargs: float | int | str | None,
    ) -> Tracks:
        """
        Runs the Hodges tracking algorithm.
        Supports chunked detection if max_chunk_size is provided. Detections are
        gathered before a single linking pass so chunk boundaries do not affect
        the result.

        Args:
            infile: Path to the input data file.
            varname: Variable name to track.
            start_time, end_time: Time range for tracking.
            mode: Search for 'min' or 'max' extrema.
            backend: Processing backend (serial, mpi, dask).
            n_workers: Number of parallel workers.
            max_chunk_size: Number of steps per time chunk.
            threshold: Intensity threshold for detection.
            engine: Data loading engine (netcdf4, h5netcdf, etc).
            overlap: Retained for cross-tracker API compatibility. Hodges gathers
                detections before linking and does not require overlap.
            min_points: Minimum grid points per object.
            filter: If True, apply spectral filtering.
            lmin, lmax: Spectral truncation range (default T5-42).
            taper_points: Boundary tapering points.
        """
        import timeit

        if backend != "serial":
            raise NotImplementedError(
                "HodgesTracker currently supports only the serial backend."
            )
        if max_chunk_size is not None and max_chunk_size < 1:
            raise ValueError("max_chunk_size must be positive")

        t_total_start = timeit.default_timer()

        # 1. Load and optionally filter data
        t0 = timeit.default_timer()
        if isinstance(infile, (xr.DataArray, xr.Dataset)):
            data_xr = infile
            if isinstance(data_xr, xr.Dataset):
                data_xr = data_xr[varname]
        else:
            detector_peek = HodgesDetector(infile, varname, engine=engine)
            if start_time is None or end_time is None:
                full_times = detector_peek.get_time()
                if start_time is None:
                    start_time = full_times[0]
                if end_time is None:
                    end_time = full_times[-1]

            data_xr = detector_peek.get_xarray(start_time, end_time)

        if filter or map_proj != "global":
            data_xr = self.preprocess_standard_track(
                data_xr,
                lmin=lmin if filter else 0,
                lmax=lmax,
                taper_points=taper_points,
                map_proj=map_proj,
                resolution=resolution,
                extent=extent,
            )
        t1 = timeit.default_timer()
        print(f"    [Serial] Preprocessing time: {t1 - t0:.4f}s")

        if max_chunk_size is None:
            tracks = self._track_single_chunk_from_data(
                data_xr,
                mode,
                threshold,
                min_points=min_points,
                subgrid_refine=subgrid_refine,
                **kwargs,
            )
        else:
            # Detection can be partitioned, but linking must see the full series.
            from ..io.data_loader import DataLoader

            time_dim = DataLoader(data_xr).get_coords()[0]
            n_steps = data_xr.sizes[time_dim]
            detections: list[RawDetectionStep] = []

            for start_idx in range(0, n_steps, max_chunk_size):
                end_idx = min(start_idx + max_chunk_size, n_steps)
                chunk_data = data_xr.isel({time_dim: slice(start_idx, end_idx)})
                detections.extend(
                    self._detect_single_chunk_from_data(
                        chunk_data,
                        mode,
                        threshold,
                        min_points=min_points,
                        subgrid_refine=subgrid_refine,
                        **kwargs,
                    )
                )
            tracks = self._link_detections(detections)

        t_total_end = timeit.default_timer()
        print(f"Tracking time: {t_total_end - t_total_start:.4f}s")
        tracks.track_type = varname
        return tracks

    def _track_single_chunk_from_data(
        self,
        data: xr.DataArray,
        mode: Literal["min", "max"] = "min",
        threshold: float | None = None,
        min_points: int = constants.MIN_POINTS_DEFAULT,
        subgrid_refine: bool = True,
        **kwargs: float | int | str | None,
    ) -> Tracks:
        detections = self._detect_single_chunk_from_data(
            data,
            mode,
            threshold,
            min_points=min_points,
            subgrid_refine=subgrid_refine,
            **kwargs,
        )
        return self._link_detections(detections)

    def _detect_single_chunk_from_data(
        self,
        data: xr.DataArray,
        mode: Literal["min", "max"] = "min",
        threshold: float | None = None,
        min_points: int = constants.MIN_POINTS_DEFAULT,
        subgrid_refine: bool = True,
        **kwargs: float | int | str | None,
    ) -> list[RawDetectionStep]:
        import timeit

        # 1. Detection
        t_detect_start = timeit.default_timer()
        detector = HodgesDetector.from_xarray(data)

        size = int(kwargs.get("size", 5))  # type: ignore[arg-type]

        detections = detector.detect(
            size=size,
            threshold=threshold,
            minmaxmode=mode,
            min_points=min_points,
            subgrid_refine=subgrid_refine,
        )

        map_proj = data.attrs.get("map_proj", "global")
        if map_proj in ("nh_stereo", "sh_stereo"):
            from ..models.geo import stereo_to_latlon

            hemi = 1 if map_proj == "nh_stereo" else -1
            converted_detections = []
            for dt, lats, lons, values in detections:
                new_lats = np.zeros_like(lats)
                new_lons = np.zeros_like(lons)
                for i in range(len(lats)):
                    # Note: lons[i] is x, lats[i] is y
                    lat, lon = stereo_to_latlon(lons[i], lats[i], hemi)
                    new_lats[i] = lat
                    new_lons[i] = lon
                converted_detections.append((dt, new_lats, new_lons, values))
            detections = converted_detections

        t_detect_end = timeit.default_timer()
        print(f"    [Serial] Detection time: {t_detect_end - t_detect_start:.4f}s")

        return detections

    def _link_detections(self, detections: list[RawDetectionStep]) -> Tracks:
        import timeit

        # Linking uses the MGE cost function with adaptive constraints.
        # Cost = w1 * (1 - cos(theta)) + w2 * (1 - 2*sqrt(d1*d2)/(d1+d2))
        # This penalizes both changes in direction and changes in speed.
        t_link_start = timeit.default_timer()
        linker = HodgesLinker(
            w1=self.w1,
            w2=self.w2,
            dmax=self.dmax,
            phimax=self.phimax,
            n_iterations=self.n_iterations,
            max_missing=self.max_missing,
            zones=self.zones,
            adapt_params=self.adapt_params,
        )

        tracks = linker.link(detections)
        t_link_end = timeit.default_timer()
        print(f"    [Serial] Linking time: {t_link_end - t_link_start:.4f}s")

        # Pruning
        valid_tracks = []
        for tr in tracks:
            if len(tr) >= self.min_lifetime:
                valid_tracks.append(tr)

        out = Tracks()
        for tr in valid_tracks:
            out.append(tr)

        return out

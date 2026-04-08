from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr

from ..hodges import constants
from ..models import TimeRange, Tracks
from ..models.tracker import RawDetectionStep
from .detector import SimpleDetector
from .linker import SimpleLinker

if TYPE_CHECKING:
    from ..models.geo import MapExtent


def _link_centers(
    raw_steps: list[RawDetectionStep], time_range: TimeRange | None = None
) -> Tracks:
    """Sequentially links raw detection steps into a global Tracks object."""
    tracks = Tracks()
    if time_range:
        tracks.time_range = time_range
    linker = SimpleLinker()
    for step_data in raw_steps:
        linker.append(tracks, step_data)
    return tracks


def _detect_and_link(
    detector: SimpleDetector,
    size: int,
    threshold: float | None,
    mode: Literal["min", "max"],
) -> list[RawDetectionStep]:
    """Worker task: Detects centers and returns raw results for central linking."""
    return detector.detect(
        size=size,
        threshold=threshold,
        minmaxmode=mode,
    )


class SimpleTracker:
    """
    A tracker implementing the PyStormTracker simple parallel algorithm.
    """

    def preprocess_standard_track(
        self,
        data: xr.DataArray,
        lmin: int = constants.LMIN_DEFAULT,
        lmax: int = constants.LMAX_DEFAULT,
        taper_points: int = constants.TAPER_DEFAULT,
        map_proj: Literal["global", "nh_stereo", "sh_stereo", "healpix"] = "global",
        resolution: float = 100.0,
        extent: MapExtent | None = None,
    ) -> xr.DataArray:
        """
        Applies standard spectral preprocessing using ducc0.
        Optionally regrids to a Polar Stereographic or HEALPix projection.
        """
        from ..preprocessing.spectral import SpectralFilter
        from ..preprocessing.taper import TaperFilter

        # Ensure data is loaded into memory for spectral filtering
        if data.chunks:
            data = data.compute()

        from typing import cast

        # 1. Tapering
        if taper_points > 0:
            taper = TaperFilter(n_points=taper_points)
            data = cast(xr.DataArray, taper.filter(data))

        # 2. Regridding and Filtering
        if map_proj in ("nh_stereo", "sh_stereo", "healpix"):
            from ..preprocessing.regrid import SpectralRegridder

            regridder = SpectralRegridder(lmax=lmax)

            # We process frame by frame
            from ..io.data_loader import DataLoader

            loader = DataLoader(data.dataset if hasattr(data, "dataset") else data)
            is_lat_reversed = loader.is_lat_reversed()

            time_dim = next(
                (c for c in DataLoader.VAR_MAPPING["time"] if c in data.dims), "time"
            )

            out_frames = []
            for i in range(len(data[time_dim])):
                frame = data.isel({time_dim: i}).squeeze()
                if map_proj == "healpix":
                    nside = int(np.sqrt(12 * (lmax + 1) ** 2 / 12))
                    nside = 2 ** int(np.round(np.log2(max(1, nside))))
                    if lmin > 0:
                        spectral_filter = SpectralFilter(lmin=lmin, lmax=lmax)
                        frame = spectral_filter.filter(frame)
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
            # Global grid filtering
            spectral_filter = SpectralFilter(lmin=lmin, lmax=lmax)
            data = spectral_filter.filter(data)

        return data

    def _detect_serial(
        self,
        infile: str,
        varname: str,
        time_range: TimeRange | None,
        mode: Literal["min", "max"],
        threshold: float | None = None,
        engine: str | None = None,
        filter: bool = True,
        lmin: int = constants.LMIN_DEFAULT,
        lmax: int = constants.LMAX_DEFAULT,
        taper_points: int = constants.TAPER_DEFAULT,
        map_proj: Literal["global", "nh_stereo", "sh_stereo", "healpix"] = "global",
        resolution: float = 100.0,
        extent: MapExtent | None = None,
        **kwargs: float | int | str | None,
    ) -> Tracks:
        import timeit

        t0 = timeit.default_timer()
        detector_peek = SimpleDetector(
            pathname=infile, varname=varname, time_range=time_range, engine=engine
        )
        data_xr = detector_peek.get_xarray()

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
        t_pre = timeit.default_timer()
        print(f"    [Serial] Preprocessing time: {t_pre - t0:.4f}s")

        t0_detect = timeit.default_timer()
        detector = SimpleDetector.from_xarray(data_xr)
        size = int(kwargs.get("size", 5))  # type: ignore[arg-type]
        raw_steps = _detect_and_link(
            detector, size=size, threshold=threshold, mode=mode
        )

        effective_map_proj = kwargs.get("map_proj", map_proj)
        if effective_map_proj in ("nh_stereo", "sh_stereo"):
            from ..models.geo import stereo_to_latlon

            hemi = 1 if effective_map_proj == "nh_stereo" else -1
            converted_raw_steps = []
            for dt, lats, lons, values in raw_steps:
                new_lats = np.zeros_like(lats)
                new_lons = np.zeros_like(lons)
                for i in range(len(lats)):
                    # Note: lons[i] is x, lats[i] is y
                    lat, lon = stereo_to_latlon(lons[i], lats[i], hemi)
                    new_lats[i] = lat
                    new_lons[i] = lon
                converted_raw_steps.append((dt, new_lats, new_lons, values))
            raw_steps = converted_raw_steps

        t1 = timeit.default_timer()
        print(f"    [Serial] Detection time: {t1 - t0_detect:.4f}s")

        t2 = timeit.default_timer()
        tracks = _link_centers(raw_steps, time_range=detector_peek.time_range)
        t3 = timeit.default_timer()
        print(f"    [Serial] Linking time: {t3 - t2:.4f}s")
        return tracks

    def track(
        self,
        infile: str,
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
        overlap: int = 3,
        min_points: int = 1,
        filter: bool = True,
        lmin: int = constants.LMIN_DEFAULT,
        lmax: int = constants.LMAX_DEFAULT,
        taper_points: int = constants.TAPER_DEFAULT,
        **kwargs: float | int | str | None,
    ) -> Tracks:
        import timeit

        t0 = timeit.default_timer()

        time_range = None
        if start_time is not None or end_time is not None:
            st = np.datetime64(start_time) if start_time else None
            et = np.datetime64(end_time) if end_time else None

            if st is None:
                st = np.datetime64("NaT")
            if et is None:
                et = np.datetime64("NaT")

            time_range = TimeRange(start=st, end=et)

        if backend == "mpi":
            from .concurrent import run_simple_mpi

            tracks = run_simple_mpi(
                infile,
                varname,
                time_range,
                mode,
                threshold=threshold,
                engine=engine,
                filter=filter,
                lmin=lmin,
                lmax=lmax,
                taper_points=taper_points,
                map_proj=map_proj,
                **kwargs,
            )
        elif backend == "dask":
            from .concurrent import run_simple_dask

            tracks = run_simple_dask(
                infile,
                varname,
                time_range,
                mode,
                n_workers,
                max_chunk_size=max_chunk_size,
                threshold=threshold,
                engine=engine,
                filter=filter,
                lmin=lmin,
                lmax=lmax,
                taper_points=taper_points,
                **kwargs,
            )
        else:
            tracks = self._detect_serial(
                infile,
                varname,
                time_range,
                mode,
                threshold=threshold,
                engine=engine,
                filter=filter,
                lmin=lmin,
                lmax=lmax,
                taper_points=taper_points,
                map_proj=map_proj,
                resolution=resolution,
                extent=extent,
                **kwargs,
            )

        t_end = timeit.default_timer()
        rank = 0
        if backend == "mpi":
            from mpi4py import MPI

            rank = MPI.COMM_WORLD.Get_rank()

        if rank == 0:
            print(f"Tracking time: {t_end - t0:.4f}s")

        tracks.track_type = varname
        return tracks

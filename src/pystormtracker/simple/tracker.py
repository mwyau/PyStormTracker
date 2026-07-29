from __future__ import annotations

from pathlib import Path
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
    subgrid_refine: bool = False,
) -> list[RawDetectionStep]:
    """Worker task: Detects centers and returns raw results for central linking."""
    return detector.detect(
        size=size,
        threshold=threshold,
        minmaxmode=mode,
        subgrid_refine=subgrid_refine,
    )


def _convert_stereo_steps(
    raw_steps: list[RawDetectionStep],
    map_proj: Literal["global", "nh_stereo", "sh_stereo", "healpix"],
) -> list[RawDetectionStep]:
    """Convert projected detection coordinates back to latitude and longitude."""
    if map_proj not in ("nh_stereo", "sh_stereo"):
        return raw_steps

    from ..models.geo import stereo_to_latlon

    hemisphere = 1 if map_proj == "nh_stereo" else -1
    converted: list[RawDetectionStep] = []
    for time, y, x, values in raw_steps:
        lats = np.empty_like(y)
        lons = np.empty_like(x)
        for i in range(len(y)):
            lats[i], lons[i] = stereo_to_latlon(x[i], y[i], hemisphere)
        converted.append((time, lats, lons, values))
    return converted


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
        filter_type: Literal["sht", "dct", "auto"] = "auto",
    ) -> xr.DataArray:
        """
        Applies standard spectral preprocessing using SHT or DCT.
        Optionally regrids to a Polar Stereographic or HEALPix projection.
        """
        # Identify spatial dimensions for auto-detection
        from ..io.data_loader import DataLoader
        from ..preprocessing.spectral import DCTFilter, SHTFilter
        from ..preprocessing.taper import TaperFilter

        loader = DataLoader(data.dataset if hasattr(data, "dataset") else data)
        _time_dim, _lat_dim, _lon_dim = loader.get_coords()

        if filter_type == "auto":
            filter_type = "sht" if loader.is_global_longitude() else "dct"

        # Ensure data is loaded into memory for spectral filtering
        if data.chunks:
            data = data.compute()

        from typing import cast

        # 1. Tapering (Spatial domain boundary tapering)
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
                    nside = int(np.sqrt(12 * (lmax + 1) ** 2 / 12))
                    nside = 2 ** int(np.round(np.log2(max(1, nside))))
                    if lmin > 0:
                        # Global projection (healpix) always uses SHT
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
            # Global or regional grid filtering (no projection)
            f_cls = SHTFilter if filter_type == "sht" else DCTFilter
            spectral_filter = f_cls(lmin=lmin, lmax=lmax)
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
        subgrid_refine: bool = False,
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
            detector,
            size=size,
            threshold=threshold,
            mode=mode,
            subgrid_refine=subgrid_refine,
        )
        raw_steps = _convert_stereo_steps(raw_steps, map_proj)

        t1 = timeit.default_timer()
        print(f"    [Serial] Detection time: {t1 - t0_detect:.4f}s")

        # 3. Linking Phase: Combine points into trajectories.
        # This implementation uses fast nearest-neighbor search based on a
        # simple distance threshold, with no recursive optimization.
        t2 = timeit.default_timer()
        tracks = _link_centers(raw_steps, time_range=detector_peek.time_range)
        t3 = timeit.default_timer()
        print(f"    [Serial] Linking time: {t3 - t2:.4f}s")
        return tracks

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
        overlap: int = 3,
        min_points: int = 1,
        filter: bool = True,
        lmin: int = constants.LMIN_DEFAULT,
        lmax: int = constants.LMAX_DEFAULT,
        taper_points: int = constants.TAPER_DEFAULT,
        subgrid_refine: bool = False,
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

        if isinstance(infile, (xr.DataArray, xr.Dataset)):
            if backend != "serial":
                msg = (
                    "Dask and MPI backends for SimpleTracker require a file path, "
                    "not an xarray object."
                )
                raise NotImplementedError(msg)

            data_xr = infile
            if isinstance(data_xr, xr.Dataset):
                data_xr = data_xr[varname]

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

            detector = SimpleDetector.from_xarray(data_xr)
            size = int(kwargs.get("size", 5))  # type: ignore[arg-type]
            raw_steps = _detect_and_link(
                detector,
                size=size,
                threshold=threshold,
                mode=mode,
                subgrid_refine=subgrid_refine,
            )
            raw_steps = _convert_stereo_steps(raw_steps, map_proj)

            tracks = _link_centers(raw_steps, time_range=time_range)

        elif backend == "mpi":
            from .concurrent import run_simple_mpi

            tracks = run_simple_mpi(
                str(infile),
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
                subgrid_refine=subgrid_refine,
                **kwargs,
            )
        elif backend == "dask":
            from .concurrent import run_simple_dask

            tracks = run_simple_dask(
                str(infile),
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
                map_proj=map_proj,
                resolution=resolution,
                extent=extent,
                subgrid_refine=subgrid_refine,
                **kwargs,
            )
        else:
            tracks = self._detect_serial(
                str(infile),
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
                subgrid_refine=subgrid_refine,
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

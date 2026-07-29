from __future__ import annotations

import timeit
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ..hodges import constants
from ..models import TimeRange, Tracks
from ..models.tracker import RawDetectionStep, Tracker
from ..preprocessing.spectral import SHTFilter
from ..preprocessing.taper import TaperFilter
from .detector import HealpixDetector

if TYPE_CHECKING:
    from ..models.geo import MapExtent


def _detect_and_gather(
    detector: HealpixDetector,
    threshold: float | None,
    mode: Literal["min", "max"],
    min_points: int,
    subgrid_refine: bool,
) -> list[RawDetectionStep]:
    """Worker task: Detects centers on HEALPix and returns raw results."""
    return detector.detect(
        threshold=threshold,
        minmaxmode=mode,
        min_points=min_points,
        subgrid_refine=subgrid_refine,
    )


class HealpixTracker(Tracker):
    """
    A tracker specifically designed for 1D HEALPix grids.
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
    ) -> xr.DataArray:
        """
        Apply spectral filtering and convert regular grids to HEALPix.
        """
        if data.chunks:
            data = data.compute()

        # 1. Tapering - Note: Tapering might need adjustment for 1D maps
        # if not using a 2D source.
        # But here we assume data might be regridded 2D -> 1D.
        if taper_points > 0:
            taper = TaperFilter(n_points=taper_points)
            data = cast(xr.DataArray, taper.filter(data))

        if data.ndim == 3:
            from ..io.data_loader import DataLoader
            from ..preprocessing.regrid import SpectralRegridder

            loader = DataLoader(data)
            time_dim, _, _ = loader.get_coords()
            lat_reverse = loader.is_lat_reversed()
            nside_estimate = max(1, lmax + 1)
            nside = 2 ** int(np.round(np.log2(nside_estimate)))
            regridder = SpectralRegridder(lmax=lmax)

            frames: list[xr.DataArray] = []
            for index in range(data.sizes[time_dim]):
                frame = data.isel({time_dim: index}).squeeze()
                if lmin > 0:
                    frame = SHTFilter(lmin=lmin, lmax=lmax).filter(frame)
                frames.append(
                    regridder.to_healpix(
                        frame,
                        nside=nside,
                        lat_reverse=lat_reverse,
                    )
                )
            data = xr.concat(frames, dim=data[time_dim])
            data.attrs["map_proj"] = "healpix"
            data.attrs["nside"] = nside

        return data

    def _detect_serial(
        self,
        infile: str,
        varname: str,
        time_range: TimeRange | None,
        mode: Literal["min", "max"],
        threshold: float | None = None,
        engine: str | None = None,
        min_points: int = 1,
        subgrid_refine: bool = True,
        **kwargs: float | int | str | None,
    ) -> Tracks:
        t0 = timeit.default_timer()
        detector = HealpixDetector(
            pathname=infile, varname=varname, time_range=time_range, engine=engine
        )

        raw_steps = _detect_and_gather(
            detector,
            threshold=threshold,
            mode=mode,
            min_points=min_points,
            subgrid_refine=subgrid_refine,
        )
        t1 = timeit.default_timer()
        print(f"    [Healpix] Detection time: {t1 - t0:.4f}s")

        t2 = timeit.default_timer()
        from ..hodges.linker import HodgesLinker

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
        tracks = linker.link(raw_steps)
        t3 = timeit.default_timer()
        print(f"    [Healpix] Linking time: {t3 - t2:.4f}s")
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
        subgrid_refine: bool = True,
        **kwargs: float | int | str | None,
    ) -> Tracks:

        t0 = timeit.default_timer()

        time_range = None
        if start_time is not None or end_time is not None:
            st = np.datetime64(start_time) if start_time else np.datetime64("NaT")
            et = np.datetime64(end_time) if end_time else np.datetime64("NaT")
            time_range = TimeRange(start=st, end=et)

        if backend == "serial":
            # For serial, load or extract the DataArray
            if isinstance(infile, (xr.DataArray, xr.Dataset)):
                data_xr = infile
                if isinstance(data_xr, xr.Dataset):
                    data_xr = data_xr[varname]
            else:
                detector_peek = HealpixDetector(
                    pathname=infile,
                    varname=varname,
                    time_range=time_range,
                    engine=engine,
                )
                data_xr = detector_peek.get_xarray()

            if data_xr.ndim == 3:
                data_xr = self.preprocess_standard_track(
                    data_xr,
                    lmin=lmin if filter else 0,
                    lmax=lmax,
                    taper_points=taper_points,
                )

            detector = HealpixDetector.from_xarray(data_xr)
            raw_steps = _detect_and_gather(
                detector,
                threshold=threshold,
                mode=mode,
                min_points=min_points,
                subgrid_refine=subgrid_refine,
            )
            from ..hodges.linker import HodgesLinker

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
            tracks = linker.link(raw_steps)
        else:
            msg = f"Backend '{backend}' not yet implemented for HealpixTracker."
            raise NotImplementedError(msg)

        t_end = timeit.default_timer()
        print(f"Total HEALPix tracking time: {t_end - t0:.4f}s")

        tracks.track_type = varname
        return tracks

"""Aligned Hodges-only center frame data passed to the MGE linker."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from ..models.time import TimePoint


@dataclass(frozen=True, slots=True)
class HodgesCenterFrame:
    """One Hodges input frame with aligned primary and diagnostic columns.

    Each diagnostic column shares the feature index of ``latitudes``,
    ``longitudes``, and ``values``. A ``None`` diagnostic unit means that the
    diagnostic uses the final primary-variable unit, resolved by the linker.
    """

    time: TimePoint
    latitudes: NDArray[np.float64]
    longitudes: NDArray[np.float64]
    values: NDArray[np.float64]
    diagnostics: Mapping[str, NDArray[np.float64]] = field(default_factory=dict)
    diagnostic_units: Mapping[str, str | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        latitudes = np.array(self.latitudes, dtype=np.float64, copy=True, order="C")
        longitudes = np.array(
            self.longitudes,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        values = np.array(self.values, dtype=np.float64, copy=True, order="C")
        feature_count = values.size
        if latitudes.ndim != 1 or longitudes.ndim != 1 or values.ndim != 1:
            raise ValueError("Hodges detection coordinate and value columns must be 1D")
        if latitudes.size != feature_count or longitudes.size != feature_count:
            raise ValueError("Hodges detection columns must have equal lengths")
        if set(self.diagnostics) != set(self.diagnostic_units):
            raise ValueError(
                "Hodges diagnostic columns and units must have identical keys"
            )

        normalized: dict[str, NDArray[np.float64]] = {}
        units: dict[str, str | None] = {}
        for name, column in self.diagnostics.items():
            if not name:
                raise ValueError("Hodges diagnostic names must be nonempty")
            diagnostic_values = np.array(
                column,
                dtype=np.float64,
                copy=True,
                order="C",
            )
            if diagnostic_values.ndim != 1 or diagnostic_values.size != feature_count:
                raise ValueError(
                    "Hodges diagnostic columns must be 1D and match feature count"
                )
            unit = self.diagnostic_units[name]
            if unit is not None and not unit:
                raise ValueError("Hodges diagnostic units must be nonempty")
            diagnostic_values.setflags(write=False)
            normalized[name] = diagnostic_values
            units[name] = unit

        latitudes.setflags(write=False)
        longitudes.setflags(write=False)
        values.setflags(write=False)
        object.__setattr__(self, "latitudes", latitudes)
        object.__setattr__(self, "longitudes", longitudes)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "diagnostics", MappingProxyType(normalized))
        object.__setattr__(self, "diagnostic_units", MappingProxyType(units))

    def with_feature_mask(self, mask: NDArray[np.bool_]) -> HodgesCenterFrame:
        """Return a row-preserving subset of every aligned feature column."""
        if mask.ndim != 1 or mask.size != self.values.size:
            raise ValueError("Hodges feature mask must match feature count")
        return HodgesCenterFrame(
            self.time,
            self.latitudes[mask],
            self.longitudes[mask],
            self.values[mask],
            {name: values[mask] for name, values in self.diagnostics.items()},
            self.diagnostic_units,
        )

    def _legacy_tuple(
        self,
    ) -> tuple[
        TimePoint,
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Provide the historical four-column detector view."""
        return self.time, self.latitudes, self.longitudes, self.values

    def __iter__(
        self,
    ) -> Iterator[TimePoint | NDArray[np.float64]]:
        """Iterate the historical four-column detector view."""
        return iter(self._legacy_tuple())

    def __getitem__(self, index: int) -> TimePoint | NDArray[np.float64]:
        """Index the historical four-column detector view."""
        return self._legacy_tuple()[index]

    def __len__(self) -> int:
        """Report the historical number of detector columns."""
        return 4

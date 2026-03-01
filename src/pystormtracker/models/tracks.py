from collections.abc import Iterator
from typing import Any

from .center import Center


class Tracks:
    def __init__(self) -> None:
        self._tracks: list[list[Center]] = []
        self.head: list[int] = []
        self.tail: list[int] = []
        self.tstart: Any | None = None
        self.tend: Any | None = None
        self.dt: Any | None = None

    def __getitem__(self, index: int) -> list[Center]:
        return self._tracks[index]

    def __setitem__(self, index: int, value: list[Center]) -> None:
        self._tracks[index] = value

    def __iter__(self) -> Iterator[list[Center]]:
        return iter(self._tracks)

    def __len__(self) -> int:
        return len(self._tracks)

    def append(self, obj: list[Center]) -> None:
        self._tracks.append(obj)

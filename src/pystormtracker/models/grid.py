from abc import ABCMeta, abstractmethod
from typing import Any, Literal

from .center import Center


class Grid(metaclass=ABCMeta):
    @abstractmethod
    def get_var(self, chart: int | tuple[int, int] | None = None) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_time(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_time_obj(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_lat(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_lon(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def split(self, num: int) -> list["Grid"]:
        raise NotImplementedError

    @abstractmethod
    def detect(
        self,
        size: int = 5,
        threshold: float = 0.0,
        chart_buffer: int = 400,
        minmaxmode: Literal["min", "max"] = "min",
    ) -> list[list[Center]]:
        raise NotImplementedError

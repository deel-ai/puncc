from __future__ import annotations

from collections.abc import Iterator
from typing import Any

class CalibrationContext:
    """
    Container for data required during calibration.

    At least y_pred and y_calib must be provided. 
    Additional sample-aligned quantities can be stored as keyword arguments.

    All stored values are expected to support sample-wise indexing.
    """

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        self.__dict__.update(kwargs)

    def __len__(self) -> int:
        if not self.__dict__:
            return 0
        return len(next(iter(self.__dict__.values())))

    def __iter__(self) -> Iterator[Any]:
        return iter(self.__dict__.values())

    def __getitem__(
        self,
        key,
    ) -> Any | CalibrationContext:
        if isinstance(key, str):
            return self.__dict__[key]

        return type(self)(
            **{
                name: value[key]
                for name, value
                in self.__dict__.items()
            }
        )
    
    def __delitem__(self, key: str) -> None:
        del self.__dict__[key]

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def __repr__(self) -> str:
        fields = ", ".join(self.__dict__)
        return (
            f"{type(self).__name__}"
            f"({fields})"
        )

    def from_values(
        self,
        values,
    ) -> CalibrationContext:
        return type(self)(
            **dict(
                zip(
                    self.keys(),
                    values,
                )
            )
        )
    
    def update(
        self,
        **kwargs: Any,
    ) -> CalibrationContext:
        self.__dict__.update(kwargs)
        return self
    
    def clear(self) -> CalibrationContext:
        self.__dict__.clear()
        return self
    
    def copy(self) -> CalibrationContext:
        return type(self)(
            **self.__dict__
        )
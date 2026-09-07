# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
CalibrationContext : Container for calibration data
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, KeysView, Self, ValuesView, ItemsView


class CalibrationContext:
    """
    Container for data required for calibration of model.
    Base behaviour saves y_calib and y_pred but some 
    calibration tools may require x_calib or other indicators depending on these data

    These additional sample-aligned quantities and can be stored as keyword arguments.

    All stored values are expected to support sample-wise indexing.
    """

    X_calib:Any
    y_calib:Any
    y_pred:Any
    nc_scores:Any

    def __init__(
        self,
        X_calib: Any = None,
        y_calib: Any = None,
        y_pred: Any = None,
        **kwargs: Any,
    ) -> None:
        if X_calib is not None:
            self.X_calib = X_calib

        if y_calib is not None:
            self.y_calib = y_calib

        if y_pred is not None:
            self.y_pred = y_pred

        self.__dict__.update(kwargs)

    @property
    def size(self)->int:
        """
        Returns:
            int: length of the calibration set.
        """
        if not self.__dict__:
            return 0
        return len(next(iter(self.__dict__.values())))

    def __iter__(self) -> Iterator[Any]:
        return iter(self.__dict__.values())

    def __getitem__(
        self,
        key:str|int|slice,
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
    
    def __contains__(self, key: str) -> bool:
        return key in self.__dict__

    def __delitem__(self, key: str) -> None:
        del self.__dict__[key]

    def keys(self)->KeysView[str]:
        return self.__dict__.keys()

    def values(self)->ValuesView[Any]:
        return self.__dict__.values()

    def items(self)->ItemsView[str, Any]:
        return self.__dict__.items()

    def __repr__(self) -> str:
        fields = ", ".join(self.__dict__)
        return (
            f"{type(self).__name__}"
            f"({fields})"
        )

    def from_values(
        self,
        values:Iterable[Any],
    ) -> Self:
        return type(self)(
            **dict(
                zip(
                    self.keys(),
                    values,
                    strict=True
                )
            )
        )
    
    def update(
        self,
        **kwargs: Any,
    ) -> Self:
        self.__dict__.update(kwargs)
        return self
    
    def clear(self) -> Self:
        self.__dict__.clear()
        return self
    
    def copy(self) -> Self:
        # Shallow copy of the context. The underlying data is not copied.
        return type(self)(
            **self.__dict__
        )
    
    def merge(self, other_context:CalibrationContext)->Self:
        if self.size != other_context.size:
            raise ValueError(
                f"Cannot merge CalibrationContext with different sizes: "
                f"{self.size} != {other_context.size}"
            )

        self.__dict__.update(other_context.__dict__)
        return self

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
Calibration data containers used by conformal predictors.

This module defines :class:`CalibrationContext`, a lightweight container used to store data associated with a calibration dataset.

A calibration context typically contains the calibration inputs, targets, model predictions, and nonconformity scores.
Additional sample-aligned quantities can be stored dynamically when required by specific conformal methods.

All values stored in a context are expected to be aligned along their first dimension.
This allows a context to be indexed or sliced as a whole while preserving the correspondence between calibration quantities.

``CalibrationContext`` intentionally behaves like a lightweight mapping without implementing the full mapping protocol.
Stored fields can be accessed by name, iterated over, copied, updated, sliced, and merged with another compatible context.

The container stores references to the underlying objects.
Operations such as :meth:`CalibrationContext.copy` therefore perform shallow copies and do not duplicate tensors, arrays, or other calibration data.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, KeysView, Self, ValuesView, ItemsView

from deel.puncc.typing import TensorLike
from deel.puncc.backend.keras import ops


class CalibrationContext:
    """
    Container for sample-aligned calibration data.

    A calibration context stores quantities associated with a calibration dataset, such as calibration inputs, targets, model predictions, and nonconformity scores.

    Additional method-specific quantities can be stored dynamically through keyword arguments.
    All stored values are expected to be aligned along their first dimension so that indexing the context applies the same selection to every field.

    Args:
        X_calib: Calibration inputs.
        y_calib: Calibration targets.
        y_pred: Model predictions on the calibration inputs.
        **kwargs: Additional sample-aligned calibration quantities.

    Note:
        Stored values are kept by reference. The context does not copy
        tensors, arrays, or other objects passed to it.
    """

    X_calib: Any
    y_calib: Any
    y_pred: Any
    nc_scores: Any

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
    def size(self) -> int:
        """
        Return the number of calibration samples.

        The size is inferred from the first value stored in the context.
        All stored values are therefore expected to have the same number of samples along their first dimension.

        Returns:
            The number of calibration samples, or ``0`` if the context is empty.

        Note:
            This property assumes that stored values retain a sample dimension.
        """
        if not self.__dict__:
            return 0
        return len(next(iter(self.__dict__.values())))

    def __iter__(self) -> Iterator[Any]:
        """
        Iterate over the values stored in the context.

        Yields:
            Stored calibration values in insertion order.
        """
        return iter(self.__dict__.values())

    def __getitem__(
        self,
        key:str|int|slice|TensorLike,
    ) -> Any | Self:
        """
        Access a field or index all stored calibration quantities.

        String keys access individual fields.
        Integer and slice indices are applied to every value stored in the context and return a new context containing the indexed data.

        Args:
            key: Field name, sample index, or sample slice.

        Returns:
            The value associated with a string key, or a new ``CalibrationContext`` containing the indexed values.

        Raises:
            KeyError: If a requested field name does not exist.
        """
        if isinstance(key, str):
            return self.__dict__[key]
        
        if isinstance(key, (int, slice)):
            return type(self)(
                **{
                    name: value[key]
                    for name, value in self.__dict__.items()
                }
            )

        return type(self)(
            **{
                name: ops.take(value, key, axis=0)
                for name, value in self.__dict__.items()
            }
        )
        # return type(self)(
        #     **{
        #         name: value[key]
        #         for name, value
        #         in self.__dict__.items()
        #     }
        # )
    
    def __contains__(self, key: str) -> bool:
        """
        Return whether a field is stored in the context.

        Args:
            key: Field name to look up.

        Returns:
            Whether the context contains ``key``.
        """
        return key in self.__dict__

    def __delitem__(self, key: str) -> None:
        """
        Remove a field from the context.

        Args:
            key: Name of the field to remove.

        Raises:
            KeyError: If the field does not exist.
        """
        del self.__dict__[key]

    def keys(self)->KeysView[str]:
        """
        Return a view over the stored field names.

        Returns:
            A dynamic view over the context keys.
        """
        return self.__dict__.keys()

    def values(self)->ValuesView[Any]:
        """
        Return a view over the stored calibration values.

        Returns:
            A dynamic view over the context values.
        """
        return self.__dict__.values()

    def items(self)->ItemsView[str, Any]:
        """
        Return a view over the stored fields and values.

        Returns:
            A dynamic view containing ``(name, value)`` pairs.
        """
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
        """
        Create a context with the same fields and new values.
        Values are associated with the existing field names in insertion order.

        Args:
            values: New values corresponding to the current context fields.

        Returns:
            A new context of the same type containing the provided values.

        Raises:
            ValueError: If the number of provided values differs from the number of fields stored in the context.
        """
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
        """
        Add or replace fields in the context.

        Args:
            **kwargs: Fields and values to store.

        Returns:
            The updated context itself.
        """
        self.__dict__.update(kwargs)
        return self
    
    def clear(self) -> Self:
        """
        Remove all fields from the context.

        Returns:
            The emptied context itself.
        """
        self.__dict__.clear()
        return self
    
    def copy(self) -> Self:
        """
        Create a shallow copy of the context.

        The context object and its field mapping are copied, but the underlying calibration values are shared with the original context.

        Returns:
            A shallow copy of the context.
        """
        return type(self)(
            **self.__dict__
        )
    
    def merge(self, other_context:CalibrationContext)->Self:
        """
        Merge another calibration context into this context.

        Fields from ``other_context`` are added to the current context.
        Existing fields with the same name are replaced.

        Args:
            other_context: Context whose fields should be merged into this context.

        Returns:
            The updated context itself.

        Raises:
            ValueError: If the two contexts contain different numbers of calibration samples.
        """
        if self.__dict__ and other_context.__dict__ and self.size != other_context.size:
            raise ValueError(
                f"Cannot merge CalibrationContext with different sizes: "
                f"{self.size} != {other_context.size}"
            )

        self.__dict__.update(other_context.__dict__)
        return self

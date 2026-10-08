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
    Object detection various tools (should be placed somewhere else)
"""

from collections import UserList
from dataclasses import Field, fields, replace
from numbers import Integral
from typing import Any, ClassVar, Generic, Iterator, Self, TypeVar, overload

from deel.puncc.typing import TensorLike
from deel.puncc.backend import ops

T = TypeVar("T")


class IterableDataclassMixin(Generic[T]):
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]
    item_type: ClassVar[type]

    def __len__(self) -> int:
        field = fields(self)[0]
        return len(getattr(self, field.name))

    @overload
    def __getitem__(self, idx: Integral) -> T:
        ...

    @overload
    def __getitem__(self, idx: slice | TensorLike) -> Self:
        ...

    def __getitem__(self, idx) -> T | Self:
        # keep both parts of the condition even if second one covers first one to avoid errors that may be raised by ops.tensor_type
        if not isinstance(idx, (Integral, slice)) and isinstance(idx, ops.tensor_type):
            idx = ops.tolist(idx)
        if isinstance(idx, Integral) and not isinstance(idx, bool):
            values = {
                item_field.name: value[idx]
                for field, item_field in zip(
                    fields(self),
                    fields(self.item_type),
                    strict=True,
                )
                if (value := getattr(self, field.name)) is not None
            }
            return self.item_type(**values)
        values = {
            field.name: value[idx]
            for field in fields(self)
            if (value := getattr(self, field.name)) is not None
        }
        return replace(self, **values)

    def __iter__(self) -> Iterator[T]:
        for i in range(len(self)):
            yield self[i]


class IndexableUserList(UserList[T]):
    @overload
    def __getitem__(self, idx: Integral) -> T:
        ...

    @overload
    def __getitem__(
        self,
        idx: list[Integral] | tuple[Integral, ...] | slice | TensorLike,
    ) -> Self:
        ...

    def __getitem__(self, idx):
        if isinstance(idx, Integral) and not isinstance(idx, bool):
            return self.data[idx]

        if isinstance(idx, slice):
            return type(self)(self.data[idx])

        if isinstance(idx, (list, tuple)):
            if all(isinstance(i, bool) for i in idx):
                idx = [i for i, selected in enumerate(idx) if selected]
            return type(self)([self.data[i] for i in idx])

        if isinstance(idx, ops.tensor_type):
            idx = ops.tolist(idx)

        return type(self)([self.data[i] for i in idx])
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
from dataclasses import fields, replace, Field
from typing import Any, ClassVar, Generic, Iterator, Self, SupportsIndex, TypeVar, overload

from deel.puncc.backend.keras import ops
from deel.puncc.typing import TensorLike


T = TypeVar("T")

class IterableDataclassMixin(Generic[T]):
    __dataclass_fields__: ClassVar[
        dict[str, Field[Any]]
    ]

    item_type: ClassVar[type]

    def __len__(self) -> int:
        field = fields(self)[0]
        return len(getattr(self, field.name))

    @overload
    def __getitem__( # type: ignore
        self,
        idx: int,
    ) -> T:
        ...

    @overload
    def __getitem__(
        self,
        idx: slice | TensorLike,
    ) -> Self:
        ...

    def __getitem__(self, idx) -> T | Self:
        values = {
            field.name: value[idx]
            for field in fields(self)
            if (value := getattr(self, field.name)) is not None
        }

        if isinstance(idx, int):
            return self.item_type(
                *values.values()
            )

        return replace(
            self,
            **values,
        )

    def __iter__(self) -> Iterator[T]:
        for i in range(len(self)):
            yield self[i]

class IndexableUserList(UserList[T]):
    @overload
    def __getitem__( # type: ignore
        self,
        idx: SupportsIndex,
    ) -> T:
        ...

    @overload
    def __getitem__(
        self,
        idx: list[int] | tuple[int, ...] | slice | TensorLike,
    ) -> Self:
        ...

    def __getitem__(self, idx:SupportsIndex|slice|list[int]|tuple[int,...]|TensorLike) -> T|Self:
        if isinstance(idx, ops.tensor_type):
            idx = ops.convert_to_numpy(idx).tolist()

        if isinstance(idx, (list, tuple)):
            return type(self)(
                [self.data[i] for i in idx]
            )

        return super().__getitem__(idx)
from collections import UserList
from dataclasses import fields, replace
from typing import ClassVar, Generic, Iterator, Self, TypeVar

from deel.puncc.keras import ops


T = TypeVar("T")

class IterableDataclassMixin(Generic[T]):
    item_type: ClassVar[type[T]]

    def __len__(self) -> int:
        field = fields(self)[0]
        return len(getattr(self, field.name))

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
    def __getitem__(self, idx):
        if isinstance(idx, ops.tensor_type):
            idx = ops.convert_to_numpy(idx).tolist()

        if isinstance(idx, (list, tuple)):
            return type(self)(
                [self.data[i] for i in idx]
            )

        return super().__getitem__(idx)
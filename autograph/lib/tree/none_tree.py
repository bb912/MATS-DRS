from abc import abstractmethod
from typing import List, TypeVar, Generic, Union

from autograph.lib.tree import TreeOperation

T = TypeVar("T")


class AbstractNoneTree(TreeOperation[Union[T, None]], Generic[T]):
    def node_value(self, outgoing_edges: List[Union[T, None]]) -> Union[T, None]:
        filtered = [o for o in outgoing_edges if o is not None]
        if len(filtered) == 0:
            return None
        else:
            return self._node_value(filtered)

    def leaf_value(self) -> T:
        return None

    @abstractmethod
    def _node_value(self, outgoing_edges: List[T]) -> T:
        raise NotImplemented

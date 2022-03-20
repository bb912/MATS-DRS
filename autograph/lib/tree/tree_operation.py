from abc import ABC, abstractmethod
from typing import List, TypeVar, Generic

T = TypeVar("T")


class TreeOperation(ABC, Generic[T]):

    @abstractmethod
    def leaf_value(self) -> T:
        raise NotImplemented

    @abstractmethod
    def node_value(self, outgoing_edges: List[T]) -> T:
        raise NotImplemented

    @abstractmethod
    def edge_combinator(self, edge_value: T, to_value: T) -> T:
        raise NotImplemented

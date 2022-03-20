from abc import ABC

from autograph.lib.tree import TreeOperation


class ICMTreeOperation(TreeOperation[float], ABC):
    def __init__(self, discount: float):
        self.discount = discount

    def edge_combinator(self, edge_value: float, to_value: float) -> float:
        return edge_value + (self.discount * to_value)

    def leaf_value(self) -> float:
        return 0.0

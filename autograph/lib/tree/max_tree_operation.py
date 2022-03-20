from typing import List

from autograph.lib.tree.none_tree import AbstractNoneTree


class MaxTreeOperation(AbstractNoneTree[float]):
    def _node_value(self, outgoing_edges: List[float]) -> float:
        return max(outgoing_edges)

    def edge_combinator(self, edge_value: float, to_value: float) -> float:
        return max(edge_value, to_value)

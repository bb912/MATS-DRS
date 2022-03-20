from typing import List

from autograph.lib.tree import TreeOperation


class SigmaTreeOperation(TreeOperation[float]):
    def leaf_value(self) -> float:
        return 1

    def node_value(self, outgoing_edges: List[float]) -> float:
        return sum(outgoing_edges) / len(outgoing_edges)

    def edge_combinator(self, edge_value: float, to_value: float) -> float:
        return to_value

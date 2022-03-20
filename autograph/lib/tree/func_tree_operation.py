from typing import List, Callable

from autograph.lib.tree.icm_tree_operation import ICMTreeOperation


class FuncTreeOperation(ICMTreeOperation[float]):

    def __init__(self, func: Callable[[List[float]], float], discount: float):
        super().__init__(discount)
        self.func = func

    def node_value(self, outgoing_edges: List[float]) -> float:
        return self.func(outgoing_edges)

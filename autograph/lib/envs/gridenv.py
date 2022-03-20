from typing import Tuple, Callable

from autograph.lib.util import element_add


class GridEnv:
    def __init__(self, shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (len(shape) == 2)

        self.shape = shape  #: size of maze

    def _in_bounds(self, place):
        """
        Is a space inside the walls of the maze?
        """
        if len(place) != len(self.shape):
            return False

        for i in range(len(place)):
            if place[i] < 0 or place[i] >= self.shape[i]:
                return False

        return True

    def _neighbors(self, place, offsets=((0, -1), (-1, 0), (0, 1), (1, 0))):
        """
        Get iterator of all in-bounds neighbors of a cell
        """
        for i in offsets:
            el = element_add(place, i)
            if self._in_bounds(el):
                yield el

    def _render(self, render_func: Callable[[int, int], Tuple[str, bool, bool]], width_per_cell: int):
        out = []
        for y in range(self.shape[1]):
            next = []
            for x in range(self.shape[0]):
                cell_inside, fill_top, fill_left = render_func(x, y)
                fill_top |= (y == 0)  # Always fill in outside grid edge
                fill_left |= (x == 0)
                out.append("+")
                top_marker = "-" if fill_top else " "
                out.append(top_marker * width_per_cell)

                left_marker = "|" if fill_left else " "
                next.append(left_marker)
                next.append(cell_inside)

            out.append("+\n")
            out.extend(next)
            out.append("|\n")

        for x in range(self.shape[0]):
            out.append("+" + ("-" * width_per_cell))

        out.append("+\n")

        return "".join(out)

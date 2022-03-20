import torch

from autograph.lib.util.stats import StatsTracker, MovingStatsTracker


def element_add(a, b):
    """
    Add two tuples, elementwise
    """
    return tuple(x + y for x, y in zip(a, b))


def element_neg(a):
    """
    Negate each element in a tuple
    """
    return tuple(-x for x in a)


def const_plane(shape, val):
    result = torch.full(shape, val)
    return result


def one_hot(num, max):
    val = torch.zeros((max,))
    val[num] = 1
    return val


def one_hot_multi(tens: torch.Tensor, max, device="cpu"):
    val: torch.Tensor = torch.zeros((*tens.shape, max), dtype=torch.uint8, device=device)
    return val.scatter_(dim=-1, index=tens.long().unsqueeze(-1), value=1)

import torch
import torch.nn as nn
from torch.nn import functional as F


def arrayify(x):
    if isinstance(x, torch.Tensor):
        return x.unsqueeze(0).float()
    else:
        return torch.tensor([x]).float()


def tuple_arrayify(x):
    return tuple(arrayify(a) for a in x)


def single_batch(func):
    def ret_func(self, state, action, next_state):
        return func(self,
                    arrayify(state),
                    arrayify(action),
                    arrayify(next_state)).squeeze()

    return ret_func


def single_batch_tuplify(func):
    def ret_func(self, state, action, next_state):
        ret = func(self, (state,), (action,), (next_state,))
        try:
            return ret[0]
        except IndexError:
            return ret

    return ret_func


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def l2_norm_squared(vec1, vec2):
    return F.mse_loss(vec1, vec2, reduction="none").sum(dim=-1)

from typing import List, Callable, Any

import torch
from torch import Tensor

from autograph.lib.util.trace_return_step import trs_fields, TraceReturnStep


class TraceInfoPreprocessor:
    def __init__(self, trace: List[TraceReturnStep], state_rewriter: Callable[[Any], Tensor], device):
        self.trace = trace
        self.state_rewriter = state_rewriter
        self.device = device

        self.cache = dict()

    def __call__(self, field_name):
        assert field_name in trs_fields

        if field_name == "state" or field_name == "next_state":
            info = tuple(self.state_rewriter(getattr(t, field_name)) for t in self.trace)
        else:
            info = tuple(getattr(t, field_name) for t in self.trace)

        if len(info) > 0 and isinstance(info[0], Tensor):
            info = torch.stack(info, dim=0)

        return to_device_tensor(info, self.device)


def to_device_tensor(arr, device):
    """
    Put tensors in the correct place and data type
    :param arr:
    :return:
    """
    return torch.as_tensor(arr, device=device, dtype=torch.float).detach()


def to_device_tensors(arrs, device):
    return tuple(to_device_tensor(a, device) for a in arrs)


def extract(tuplelist, fieldname: str, device):
    """Shortcut to map over fields of a tuple and aggregate into a tensor."""
    return to_device_tensor(tuple(getattr(t, fieldname) for t in tuplelist), device)


def extract_sas_from_trace(net, trace, device):
    """
    Given a trace, get the state, action, state pairs in the format that the mazenet and icm can use
    """
    s, a, ns = zip(*[(net.rewrite_obs(step.state),
                      step.action_selected,
                      net.rewrite_obs(step.next_state))
                     for step in trace])

    return to_device_tensor(torch.stack(s, dim=0), device), \
           to_device_tensor(a, device), \
           to_device_tensor(torch.stack(ns, dim=0), device)

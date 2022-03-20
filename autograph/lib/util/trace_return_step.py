from collections import namedtuple

ts_fields = ("state", "value", "action_probs", "action_selected", "next_state", "next_value", "reward", "info", "done")
trs_fields = (
    "state", "value", "action_probs", "action_selected", "next_state", "next_value", "reward", "info", "done",
    "discounted_return", "advantage", "return_advantage")
TraceStep = namedtuple("TraceStep", ts_fields)
TraceReturnStep = namedtuple("TraceReturnStep", trs_fields)

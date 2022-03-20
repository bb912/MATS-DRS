from typing import Callable, Tuple, Dict, Union

import ptan
import torch
from torch import Tensor
from torch.nn import Module, functional as F

from autograph.lib.util import one_hot_multi
from autograph.lib.util.trace_info_processor import TraceInfoPreprocessor

LossFunction = Callable[[TraceInfoPreprocessor], Tuple[Tensor, Dict[str, Union[float, Tensor]]]]


class PPOLossFunction:
    def __init__(self, net: Module, device, discount: float, epsilon: float = .2, sync_old_net_every: int = 10,
                 **kwargs):
        self.discount = discount
        self.device = device
        self.epsilon = epsilon
        self.net = net
        self.old_net = ptan.agent.TargetNet(net)
        self.sync_every = sync_old_net_every
        self.sync_counter = 0

    def __call__(self, trace_info: TraceInfoPreprocessor):
        self.sync_counter += 1
        if self.sync_counter % self.sync_every == 0:
            self.sync_counter = 0
            self.old_net.sync()

        states = trace_info("state")
        actions = trace_info("action_selected")
        rewards = trace_info("reward")
        values = trace_info("value")
        ns = trace_info("next_state")

        net_policy, net_values = self.net(states)
        net_policy_probs = F.softmax(net_policy, dim=-1)

        _, next_net_values = self.net(ns)
        advantages = rewards + (self.discount * next_net_values) - net_values

        old_net_policy, _ = self.old_net.target_model(states)
        old_net_policy_probs = F.softmax(old_net_policy, dim=-1).detach()

        action_space = net_policy_probs.size(-1)
        actions_one_hot = one_hot_multi(actions, action_space, self.device)

        old_net_policy_probs_selected = torch.masked_select(old_net_policy_probs, actions_one_hot)
        net_policy_probs_selected = torch.masked_select(net_policy_probs, actions_one_hot)

        # Add the tiny amount in case probability dropped near zero
        ratio = (net_policy_probs_selected + 1e-6) / (old_net_policy_probs_selected + 1e-6)
        surr_obj = advantages * ratio
        clipped_surr_obj = advantages * torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        policy_loss = -torch.min(surr_obj, clipped_surr_obj).mean()

        value_loss = F.mse_loss((values + advantages), net_values)

        agent_loss = value_loss + policy_loss

        return agent_loss, {
            "loss/value_loss": value_loss,
            "loss/policy_loss": policy_loss,
            "loss/agent_loss": agent_loss
        }


class AdvantageActorCriticLossFunction:
    def __init__(self, net: Module, discount: float, device, **kwargs):
        self.net = net
        self.discount = discount
        self.device = device

    def __call__(self, trace_info: TraceInfoPreprocessor):
        states = trace_info("state")
        actions = trace_info("action_selected")
        values = trace_info("value")
        ns = trace_info("next_state")
        rewards = trace_info("reward")
        orig_action_probs = trace_info("action_probs")
        total_return = trace_info("discounted_return")

        net_policy, net_values = self.net(states)
        # _, next_net_values = self.net(ns)

        advantages = total_return - net_values

        net_policy_logprobs = -F.log_softmax(net_policy, dim=-1)
        # softmax: 0 to 1
        # log: -inf to 0
        # net_policy_logprobs: 0 to inf (reversed on policy)

        value_loss = F.mse_loss(total_return, net_values)
        action_space = net_policy.size(-1)
        actions_one_hot = one_hot_multi(actions, action_space, self.device)

        net_policy_logprobs_selected = torch.masked_select(net_policy_logprobs, actions_one_hot)

        importance = torch.masked_select(F.softmax(net_policy, dim=-1) / F.softmax(orig_action_probs, dim=-1),
                                         actions_one_hot)

        policy_loss = (advantages.detach() * importance.detach() * net_policy_logprobs_selected).mean()
        # Positive advantage, likely action: low positive
        # Positive advantage, rare action: high positive
        # Negative advantage, likely action: low negative
        # Negative advantage, rare action: high negative
        # So we want to decrease this amount

        agent_loss = value_loss + policy_loss

        # TODO anchor value to 0 if done is true

        return agent_loss, {
            "loss/value_loss": value_loss,
            "loss/policy_loss": policy_loss,
            "loss/agent_loss": agent_loss
        }


class TakeSimilarActionsLossFunction:
    def __init__(self, net: Module, device, no_adv=False, **kwargs):
        self.device = device
        self.net = net
        self.no_adv = no_adv

    def __call__(self, trace_info):
        disc_return = trace_info("discounted_return")
        values = trace_info("value")

        states = trace_info("state")
        actions = trace_info("action_selected")

        net_policy, net_values = self.net(states)

        # why is discounted return needed for adv agent
        if self.no_adv:
            value_loss = F.mse_loss(values, net_values)
        else:
            value_loss = F.mse_loss(disc_return, net_values.squeeze(-1))

        policy_loss = F.cross_entropy(net_policy, actions.long())

        agent_loss = value_loss + policy_loss

        # -1 * probability * log(prob)

        entropy = (-F.log_softmax(net_policy, dim=-1) * F.softmax(net_policy, dim=-1)).sum(dim=-1).mean()

        # entropy_loss = (entropy - log(6))^2
        # loss + (const * entropy_loss)

        return agent_loss, {
            "loss/value_loss": value_loss,
            "loss/policy_loss": policy_loss,
            "loss/agent_loss": agent_loss,
            "loss/entropy": entropy
        }

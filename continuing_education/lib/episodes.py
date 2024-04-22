from abc import ABC, abstractmethod
from argparse import Action
from dataclasses import dataclass
from sre_parse import State

from gym import Env
import torch
from torch import nn

from continuing_education.lib.types import Done, LogProb, Reward

from typing import Generator


@dataclass
class SARSA:
    """This is a unified dataclass for collect_actions. It has all the information almost any method would need."""

    state: State
    action: Action
    reward: Reward
    next_state: State
    done: Done = Done(False)
    next_action: Action | None = None
    action_log_prob: LogProb = LogProb(0.0)


class DiscreteActionNetworkInterface(nn.Module, ABC):
    """This is an abstract class for a feed forward network that outputs discrete actions,
    you must implement the act method for each different type."""

    def __init__(
        self, device, *, state_size: int, action_size: int, hidden_sizes: list[int]
    ) -> None:
        """
        This is a very simple feed forward network
        with an input of size state_size, and output of size action_size
        and ReLU activations between the layers
        """
        super().__init__()
        assert len(hidden_sizes) > 0, "Need at least one hidden layer"
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_sizes = hidden_sizes
        self.device = device

        # Dimensions in the network are (batch_size, input_size, output_size)
        network: list[nn.Module] = []
        network.append(
            nn.Linear(state_size, hidden_sizes[0])
        )  # Shape: (:, state_size, hidden_sizes[0])
        network.append(nn.ReLU())
        for i in range(len(hidden_sizes) - 1):
            network.append(
                nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])
            )  # Shape: (:, hidden_sizes[i], hidden_sizes[i+1])
            network.append(nn.ReLU())
        network.append(
            nn.Linear(hidden_sizes[-1], action_size)
        )  # Shape: (:, hidden_sizes[-1], action_size)
        self.network = nn.Sequential(*network).to(self.device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Takes a state tensor and returns logits along the action space"""
        state = state.to(self.device)
        return self.network(state)

    @abstractmethod
    def act(self, state: State, **kwargs) -> tuple[Action, LogProb]:
        """This method is used to sample an action from the policy"""
        pass


def collect_episode(
    *, env: Env, policy: DiscreteActionNetworkInterface, max_t: int, **policy_kwargs
) -> Generator[SARSA, None, None]:
    """A generator that yields SARSA tuples for a single episode."""
    state, _ = env.reset()
    action, action_logprob = policy.act(state, **policy_kwargs)
    next_state, reward, done, _, _ = env.step(action)
    for _ in range(max_t):
        next_action, next_action_logprob = policy.act(next_state, **policy_kwargs)
        yield SARSA(
            state=State(state),
            action=Action(action),
            reward=Reward(reward),
            next_state=State(next_state),
            next_action=Action(next_action),
            done=done,
            action_log_prob=action_logprob,
        )
        if done:
            break
        state, action, action_logprob = next_state, next_action, next_action_logprob
        next_state, reward, done, _, _ = env.step(action)

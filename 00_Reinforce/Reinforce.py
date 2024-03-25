# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: continuing-education-vJKa4-To-py3.10
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Reinforce
#
# Tutorial: https://huggingface.co/learn/deep-rl-course/unit4/introduction

# %%
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

# %% [markdown]
# # Environment
#
# First we will create the cartpole environment.
#
# The observation_space of cartpole is a 4-dimensional float vector,
# and the action_space is a discrete space with 2 possible actions (left or right).

# %%
import gym

env = gym.make("CartPole-v1")
observation_space_shape = env.observation_space.shape
action_space_size = env.action_space.n  # type: ignore
print("State size:", observation_space_shape)
print("Action size:", action_space_size)
state = env.reset()
print(f"Example state: {state}")
action_return = env.step(1)
print(f"Action return: {action_return}")

# %% [markdown]
# # Model
#
# This is the policy network, in the paper represented by $\pi_{\theta}(s_t)$
#
# Meaning the policy $\pi$ given the parameters $\theta$ (which in this code
# represents the weights and biases of self.input, self.hidden and self.output) when
# doing a forward pass with the state $s$ at time $t$ as input.
#
# The network is very simple feed forward network, with relu activation functions and a softmax output.
#
# The output of the forward method is what the paper calls $\pi_{\theta}(a_i | s_t)$, which is a PDF due to the `softmax`.
#
# The action method is a translation from a numpy state vector into an int action, using the forward pass of the network and the REINFORCE score function.
#

# %%
from typing import NewType
import numpy.typing as npt
import numpy as np

# Lets make some types to make type annotation easier
State = NewType("State", npt.NDArray[np.float64])
Action = NewType("Action", int)
Reward = NewType("Reward", float)

# %%
from typing import List, Tuple
from torch import nn
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(
        self, state_size: int, action_size: int, hidden_sizes: List[int]
    ) -> None:
        super().__init__()
        assert len(hidden_sizes) > 0, "Need at least one hidden layer"
        network = [nn.Linear(state_size, hidden_sizes[0]), nn.ReLU()]
        for i in range(len(hidden_sizes) - 1):
            network.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            network.append(nn.ReLU())
        network.append(nn.Linear(hidden_sizes[-1], action_size))
        network.append(nn.Softmax())
        self.network = nn.Sequential(*network)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

    def act(self, state: State) -> Tuple[Action, float]:
        # First we got to convert out of numpy and into pytorch
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        print(f"state: {state}")
        print(f"state_unsqueezed: {np.expand_dims(state, axis=0)}")
        pdf = self.forward(
            state_tensor
        ).cpu()  # TODO: If softmax produces a PDF, why do we need Categorical to sample from it?
        print(f"PDF: {pdf}")
        multinomial = Categorical(
            pdf
        )  # TODO: Study up on multinomial distributions and log probs
        # TODO: Take the argmax of the pdf instead and see what happens!
        print(f"Multinomial: {multinomial}")
        action_idx = np.argmax(multinomial)  # type: ignore
        return Action(action_idx.item()), multinomial.log_prob(action_idx)


# %% [markdown]
# # Training
#
# Training is done by assembling a sample of trajectories, which are lists of tuples of (state, action, reward).

# %%
from dataclasses import dataclass


@dataclass
class SAR:
    state: State
    action: Action
    reward: Reward
    log_prob: float


Trajectory = NewType("Trajectory", List[SAR])
RewardTrajectory = NewType("RewardTrajectory", List[Reward])


# %%
def collect_episode(policy: Policy) -> Tuple[Trajectory, Reward]:
    """Returns the trajectory and the sum of all rewards."""
    state, _ = env.reset()
    done = False
    trajectory = []
    while not done:
        action, log_prob = policy.act(state)
        state, reward, done, _, _ = env.step(action)
        trajectory.append(
            SAR(
                state=State(state),
                action=action,
                reward=Reward(reward),
                log_prob=log_prob,
            )
        )
    return Trajectory(trajectory), Reward(sum(sar.reward for sar in trajectory))


# %% [markdown]
# This represents the formula $R(\tau)$ in the tutorial. It's a simple reward decay formula.

# %%
import pytest


def cumulative_return(trajectory: RewardTrajectory, gamma: float = 0.5) -> float:
    if len(trajectory) == 0:
        raise ValueError("Trajectory needs at least one item.")
    if len(trajectory) == 1:
        return 0.0
    out: float = trajectory[1]
    if len(trajectory) == 2:
        return out
    for i in range(2, len(trajectory)):
        out += gamma * trajectory[i]
        gamma *= gamma
    return out


# Its important to test equations like this!
@pytest.mark.parametrize(
    "test_input,expected",
    [([0], 0), ([1, 1], 1), ([1, 1, 1], 1.5), ([1, 1, 1, 1], 1.75)],
)
def test_cumulative_return(test_input: RewardTrajectory, expected: float) -> None:
    assert cumulative_return(test_input, gamma=0.5) == expected

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

# %%
# %load_ext autoreload
# %autoreload 2

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
from torch import Tensor

# Lets make some types to make type annotation easier
State = NewType("State", npt.NDArray[np.float64])
Action = NewType("Action", int)
Reward = NewType("Reward", float)
LogProb = NewType("LogProb", Tensor)
LogLikelihood = NewType("LogLikelihood", Tensor)

# %%
from typing import List, Tuple
from torch import nn


class Policy(nn.Module):
    """A classic policy network is one which takes in a state
    and returns a probability distribution over the action space"""

    def __init__(
        self, state_size: int, action_size: int, hidden_sizes: List[int]
    ) -> None:
        """
        This is a very simple feed forward network
        with an input of size state_size, and output of size action_size
        and ReLU activations between the layers
        """
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
        """Takes a state tensor and returns a probability distribution along the action space"""
        return self.network(state)

    def act(self, state: State) -> Tuple[Action, LogProb]:
        """Same as forward, instead of returning the entire distribution, we
        return the maximum probability action
        along with the log probability of that action
        """
        # First we got to convert out of numpy and into pytorch
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)

        # Now we can run the forward pass, whos output is a probability distribution
        # along the action space
        pdf = self.forward(state_tensor)

        # Now we want to get the action that corresponds to the highest probability
        action_idx = torch.argmax(pdf)

        # We also need the log probability of the action
        # However, we are going to do backprop through the log probability of the action
        # Therefore this needs to stay as a tensor
        # The Category distribution in torch has a method for a backprop friendly log probability of one action from a multinomial distribution
        log_prob = torch.distributions.Categorical(pdf).log_prob(action_idx)

        # We return the action and the log probability of the action
        return Action(action_idx.item()), log_prob


# %% [markdown]
# # Training - REINFORCE
#
# This is the training loop for the REINFORCE algorithm.
#
# Training is done by assembling a sample of trajectories, which are lists of tuples of (state, action, reward).
#
# The algorithm is as follows:
#
# 1. Start with policy model $\pi_{\theta}$
# 2. repeat:
#     1. Generate an episode $S_0, A_0, r_0, ..., S_{T-1}, A_{T-1}, r_{T-1}$ following $\pi_{\theta}$
#     2. for t from T-1 to 0:
#         1. $G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_k$
#     3. $L(\theta) = \frac{1}{T} \sum_{t=0}^{T-1} G_t \log \pi_{\theta}(a_t | s_t)$
#     4. Optimize $\pi_{\theta}$ using $\nabla_{\theta} L(\theta)$
#
# First lets create some helper functions and types to use in the training loop.

# %%
from dataclasses import dataclass


# SAR stands for State, Action, Reward
@dataclass
class SAR:
    state: State
    action: Action
    reward: Reward
    log_prob: LogProb


# A list of SAR representing a single episode
Trajectory = NewType("Trajectory", List[SAR])
# A list of just the rewards from a single episode
RewardTrajectory = NewType("RewardTrajectory", List[Reward])


# %%
def collect_episode(policy: Policy, max_t=1000) -> Trajectory:
    """2.1. Returns the trajectory and the sum of all rewards."""
    state, _ = env.reset()
    done = False
    trajectory = []
    for _ in range(max_t):
        action, log_prob = policy.act(state)
        state, reward, done, _, _ = env.step(action)
        trajectory.append(
            SAR(
                state=State(state),
                action=action,
                reward=Reward(reward),
                log_prob=LogProb(log_prob),
            )
        )
        if done:
            break
    return Trajectory(trajectory)


# %%
def cumulative_discounted_rewards(
    trajectory: RewardTrajectory, gamma: float = 0.5
) -> RewardTrajectory:
    """2.2.1 Returns the cumulative discounted rewards of a trajectory for each timestep."""
    if len(trajectory) == 0:
        raise ValueError("Trajectory needs at least one item.")
    if len(trajectory) == 1:
        return RewardTrajectory([trajectory[0]])
    discounted_rewards: List[Reward] = []
    cumulative_reward: Reward = Reward(0)
    for reward in reversed(trajectory):
        cumulative_reward = Reward(reward + gamma * cumulative_reward)
        discounted_rewards.append(cumulative_reward)
    return RewardTrajectory(discounted_rewards[::-1])


# %%
# It's important to test our code, so we know it works as expected
# We tried to use ipytest but it wasn't working https://github.com/chmp/ipytest
assert cumulative_discounted_rewards(
    RewardTrajectory([0]), gamma=0.5
) == RewardTrajectory([0])
assert cumulative_discounted_rewards(
    RewardTrajectory([1]), gamma=0.5
) == RewardTrajectory([1])
assert cumulative_discounted_rewards(
    RewardTrajectory([1, 1]), gamma=0.5
) == RewardTrajectory([1.5, 1])
assert cumulative_discounted_rewards(
    RewardTrajectory([1, 1, 1]), gamma=0.5
) == RewardTrajectory([1.75, 1.5, 1])


# %%
def normalize(returns: Tensor) -> Tensor:
    ## standardization of the returns is employed to make training more stable
    eps = np.finfo(np.float32).eps.item()

    ## eps is the smallest representable float, which is
    # added to the standard deviation of the returns to avoid numerical instabilities
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    return returns


# %%
def log_likelihood(policy: Policy, trajectory: Trajectory, gamma=0.5) -> LogLikelihood:
    """
    2.2.2 Returns the likelihood of a trajectory given a policy.
    Instead of doing 1/T, we normalize the cumulative discounted rewards as it says
    to do in the tutorial.
    Also we use torch.cat and sum for backprop reasons
    """
    log_likelihoods = []
    cum_disc_rewards = normalize(
        cumulative_discounted_rewards(
            RewardTrajectory([sar.reward for sar in trajectory]), gamma=gamma
        )
    )
    for cum_disc_reward, sar in zip(cum_disc_rewards, trajectory):
        _, log_prob = policy.act(sar.state)
        log_likelihoods.append(cum_disc_reward * -log_prob)
    return LogLikelihood(torch.cat(log_likelihoods).sum())


# %% [markdown]
# # Train

# %%
import torch.optim as optim

policy = Policy(
    observation_space_shape[0],
    action_space_size,
    [16],
).to(device)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)

# %%
from tqdm.notebook import trange

scores = []
for i in trange(10000):
    trajectory = collect_episode(policy)
    scores.append(sum([sar.reward for sar in trajectory]))
    policy_loss = log_likelihood(policy, trajectory, gamma=1.0)
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

# %%
import plotly.express as px

fig = px.line(scores, log_y=True, title="log Scores over time")
fig.show()

# %% [markdown]
#

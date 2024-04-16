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

# %%
from pathlib import Path

if __name__ == "__main__":
    __this_file = Path().resolve() / "reinforce.ipynb"  # jupyter does not have __file__

# %%
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    print(DEVICE)

# %% [markdown]
# # Reinforce
#
# The REINFORCE algorithm has us directly optimize a policy network $\pi_\theta(s)$ by maximizing the probability we will perform each action multiplied by the cumulative discounted future reward `of taking each action, following a training set of trajectories which it generates each iteration based on the current policy.

# %% [markdown]
# # Environment
#
# First we will create the cartpole environment.
#
# The observation_space of cartpole is a 4-dimensional float vector,
# and the action_space is a discrete space with 2 possible actions (left or right).

# %%
import gym
from typing import cast


def get_environment_space(env_name: str) -> tuple[tuple[int, ...], int]:
    env = gym.make(env_name)
    observation_space_shape = env.observation_space.shape
    assert (
        observation_space_shape is not None
    ), "Observation space shape should not be None"
    action_space_size = env.action_space.n  # type: ignore[attr-defined]
    print("State size:", observation_space_shape)
    print("Action size:", action_space_size)
    state = env.reset()
    print(f"Example state: {state}")
    action_return = env.step(1)
    print(f"Action return: {action_return}")
    return observation_space_shape, cast(int, action_space_size)


if __name__ == "__main__":
    OBSERVATION_SPACE_SHAPE, ACTION_SPACE_SIZE = get_environment_space("CartPole-v1")

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
State = NewType("State", npt.NDArray[np.float32])
Action = NewType("Action", int)
Reward = NewType("Reward", float)
LogProb = NewType("LogProb", Tensor)
Loss = NewType("Loss", Tensor)

# %%
from typing import List, Tuple
from torch import nn


def softmax_with_temperature(
    logits: Tensor, *, temperature: float = 1.0, dim: int = -1
) -> Tensor:
    """Softmax with temperature"""
    return torch.exp(logits / temperature) / torch.exp(logits / temperature).sum(
        dim=dim
    )


class Policy(nn.Module):
    """A classic policy network is one which takes in a state
    and returns a probability distribution over the action space"""

    def __init__(
        self, *, state_size: int, action_size: int, hidden_sizes: List[int]
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
        self.network = nn.Sequential(*network).to(DEVICE)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Takes a state tensor and returns logits along the action space"""
        state = state.to(DEVICE)
        return self.network(state)

    def act(self, state: State, *, temperature: float) -> Tuple[Action, LogProb]:
        """Same as forward, instead of returning the entire distribution, we
        return the maximum probability action
        along with the log probability of that action
        temperature is only here for forward compatibility with other policies
        it wont affect output since we use argmax.
        """
        # First we got to convert out of numpy and into pytorch
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)

        # Now we can run the forward pass, whos output is a probability distribution
        # along the action space
        pdf = softmax_with_temperature(
            self.forward(state_tensor), temperature=temperature
        )
        assert torch.isclose(
            pdf.sum().cpu(), torch.Tensor([1.0])
        ).all(), "The output of the network should be a probability distribution"
        assert (
            pdf.cpu().shape[-1] == self.action_size
        ), "The output of the network should be a probability distribution over the action space"

        # Now we want to get the action that corresponds to the highest probability
        # TODO: We could sample from the pdf instead of taking the greedy argmax
        action_idx = torch.argmax(pdf)

        # We also need the log probability of the action
        # However, we are going to do backprop through the log probability of the action
        # Therefore this needs to stay as a tensor
        # The Category distribution in torch has a method for a backprop friendly log probability of one action from a multinomial distribution
        log_prob = torch.distributions.Categorical(pdf).log_prob(action_idx)

        # We return the action and the log probability of the action
        return Action(int(action_idx.item())), LogProb(log_prob)


# %%
# Lets print this model architecture
if __name__ == "__main__":
    policy = Policy(state_size=4, action_size=2, hidden_sizes=[16, 16])
    print(policy.network)

# %% [markdown]
# # Training - REINFORCE
#
# Training is done by assembling a sample of trajectories from the current policy, which are lists of tuples of (state, action, reward).
#
# First lets create some types to represent these trajectories.

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


# %% [markdown]
# And here is a nice helper function I found in the hugging face tutorial for normalization in pytorch.


# %%
def normalize(returns: Tensor | list[float] | npt.NDArray[np.float32]) -> Tensor:
    """
    Standard normalizes a tensor of float32s using the mean and standard deviation.
    Handles floating point errors by adding a small epsilon to the denominator to avoid division by zero.

    $\frac{x - E[x]}_{std(x) + eps}$

    Ripped off from huggingface https://huggingface.co/learn/deep-rl-course/unit4/hands-on
    """
    ## standardization of the returns is employed to make training more stable
    eps = np.finfo(np.float32).eps.item()

    ## eps is the smallest representable float, which is
    # added to the standard deviation of the returns to avoid numerical instabilities
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    return returns


# %% [markdown]
# Now we need a way to collect an episode from the environment given a policy. That is fairly simple using gymnasium.
#
# In the hugging face tutorial, this is represented by the step which says:
#
# Generate an episode $S_0, A_0, r_0, ..., S_{T-1}, A_{T-1}, r_{T-1}$ following $\pi_{\theta}$

# %%
from gym import Env


def collect_episode(
    *, env: Env, policy: Policy, max_t: int = 1000, temperature: float = 1.0
) -> Trajectory:
    """2.1 Returns the trajectory of one episode of using the policy.

    The output is a list of SAR tuples, where each tuple represents a state, action, reward tuple.

    In the hugging face tutorial this is represented as:

    $S_0, A_0, r_0, ..., S_{T-1}, A_{T-1}, r_{T-1}$
    """
    state, _ = env.reset()
    done = False
    trajectory = []
    for _ in range(max_t):
        action, log_prob = policy.act(state, temperature=temperature)
        next_state, reward, done, _, _ = env.step(action)
        trajectory.append(
            SAR(
                state=State(state),
                action=action,
                reward=Reward(reward),
                log_prob=LogProb(log_prob),
            )
        )
        state = next_state
        if done:
            break
    return Trajectory(trajectory)


# %% [markdown]
# The above function returns the rewards at each step of the episode, but for training purposes we need the discounted future rewards for each step. This is because we want to reward each action based on the future rewards it leads to, not just the immediate reward. We discount future rewards by a factor of $\gamma$ for stability, and also because future rewards are less certain, and therefore less valuable.
#
# $G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_k$

# %%
# This is different than a reward trajectory
# because each index is the discounted sum of future indexes
# in the corresponding reward trajectory
CumDiscFutureRewardTrajectory = NewType("CumDiscFutureRewardTrajectory", list[Reward])


def cumulative_discounted_future_rewards(
    *, trajectory: RewardTrajectory, gamma: float
) -> CumDiscFutureRewardTrajectory:
    """2.2.1 Returns the cumulative discounted future rewards of a trajectory at each step of the trajectory.

    In the hugging face tutorial,
    each element in the output is represented by $G_t$ where $t$ is the index of the element."""
    if len(trajectory) == 0:
        raise ValueError("Trajectory needs at least one item.")
    if len(trajectory) == 1:
        return CumDiscFutureRewardTrajectory([trajectory[0]])
    discounted_rewards: List[Reward] = []
    cumulative_reward: Reward = Reward(0)
    for reward in reversed(trajectory):
        cumulative_reward = Reward(reward + gamma * cumulative_reward)
        discounted_rewards.append(cumulative_reward)
    return CumDiscFutureRewardTrajectory(discounted_rewards[::-1])


def test_cumulative_discounted_future_rewards() -> None:
    # It's important to test our code, so we know it works as expected
    # We tried to use ipytest but it wasn't working https://github.com/chmp/ipytest
    assert cumulative_discounted_future_rewards(
        trajectory=RewardTrajectory([Reward(-1)]), gamma=0.5
    ) == RewardTrajectory([Reward(-1.0)])
    assert cumulative_discounted_future_rewards(
        trajectory=RewardTrajectory([Reward(0)]), gamma=0.5
    ) == RewardTrajectory([Reward(0.0)])
    assert cumulative_discounted_future_rewards(
        trajectory=RewardTrajectory([Reward(0), Reward(1)]), gamma=0.5
    ) == RewardTrajectory([Reward(0.5), Reward(1.0)])
    assert cumulative_discounted_future_rewards(
        trajectory=RewardTrajectory([Reward(0), Reward(1), Reward(1)]), gamma=0.5
    ) == RewardTrajectory([Reward(0.75), Reward(1.5), Reward(1.0)])


if __name__ == "__main__":
    # I can't get pytest to work with jupyter, so I'm just going to run the tests here
    test_cumulative_discounted_future_rewards()
    print("test_cumulative_discounted_future_rewards passed")


# %% [markdown]
# Finally, this is the objective function. We want to sum the log probabilities of each action taken in the trajectory, multiplied by the discounted future rewards resulting from that action. We negate the log probabilities because we want to maximize the objective function, and the optimizer we are using is a minimizer. This is the REINFORCE score function.
#
# $J(\theta) = \frac{1}{T} \sum_{t=0}^{T-1} G_t \log \pi_{\theta}(a_t | s_t)$


# %%
def objective(*, policy: Policy, trajectory: Trajectory, gamma: float) -> Loss:
    """
    2.2.2 Returns the likelihood of a trajectory given a policy.
    Instead of doing 1/T, we normalize the cumulative discounted rewards as it says
    to do in the tutorial.
    Also we use torch.cat and sum for backprop reasons, so that backward can be called on the output.
    """
    loss = []
    cum_disc_rewards = normalize(
        cast(
            list[float],
            cumulative_discounted_future_rewards(
                trajectory=RewardTrajectory([sar.reward for sar in trajectory]),
                gamma=gamma,
            ),
        )
    )
    for cum_disc_reward, sar in zip(cum_disc_rewards, trajectory):
        loss.append(
            cum_disc_reward * -sar.log_prob
        )  # This is negative to turn maximization into minimization
    return Loss(torch.cat(loss).sum())


# %% [markdown]
# Putting it all together, this is the training loop for the REINFORCE algorithm:
#
# 1. Start with policy model $\pi_{\theta}$
# 2. repeat:
#     1. Generate an episode $S_0, A_0, r_0, ..., S_{T-1}, A_{T-1}, r_{T-1}$ following $\pi_{\theta}$
#     2. for t from T-1 to 0:
#         1. $G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_k$
#     3. $L(\theta) = \frac{1}{T} \sum_{t=0}^{T-1} G_t \log \pi_{\theta}(a_t | s_t)$
#     4. Optimize $\pi_{\theta}$ using $\nabla_{\theta} L(\theta)$
#

# %%
from tqdm.notebook import trange
import torch.optim as optim


def reinforce_train(
    *,
    env: Env,
    policy: Policy,
    optimizer: optim.Optimizer,
    gamma: float,
    num_episodes: int,
    max_t: int,
) -> list[float]:
    """Algorithm 1 REINFORCE"""
    assert gamma <= 1, "Gamma should be less than or equal to 1"
    assert gamma > 0, "Gamma should be greater than 0"
    assert num_episodes > 0, "Number of episodes should be greater than 0"
    scores: list[float] = []
    for _ in trange(num_episodes):
        # TODO: We could batch these episodes to get more stability
        trajectory = collect_episode(
            env=env, policy=policy, max_t=max_t, temperature=1.0
        )
        scores.append(sum([sar.reward for sar in trajectory]))
        policy_loss = objective(policy=policy, trajectory=trajectory, gamma=gamma)
        optimizer.zero_grad()
        policy_loss.backward()  # This gives us the gradient
        optimizer.step()
    return scores


# %% [markdown]
# ## Unit Testing NN's and Training Functions
#
#
# First we want to unit test this on the simplest environment we can possibly think of, something that if it does not work it **guarenteed** to be a coding error. Something like "reward 1 if you repeat your input, 0 otherwise, end after 10 right answers."

# %%
from gym import spaces
import random

from numpy import float32


class MockEnv(gym.Env):
    """A dead simple environment for reinforcement learning that rewards the agent for going left.
    Useful for unit testing.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, max_steps=10):
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(1)
        self.max_steps = max_steps
        self.state: npt.NDArray[np.float32] = np.array([random.choice([0.0, 1.0])])  # type: ignore[annotation-unchecked]
        self.steps = 0

    def step(
        self, action: Action
    ) -> tuple[npt.NDArray[float32], Reward, bool, bool, dict]:
        assert self.action_space.contains(action), f"Invalid Action: {action}"

        # If the state is 0, then the 0th action is the correct action
        # If the state is 1, then the 1st action is the correct action
        reward = Reward(1) if action == self.state else Reward(-1)
        self.steps += 1
        done = self.steps >= self.max_steps
        if done:
            return self.state, reward, done, False, {}
        else:
            self.state = np.array([random.choice([0.0, 1.0])])
            return self.state, reward, done, False, {}

    def reset(self) -> tuple[npt.NDArray[float32], dict]:  # type: ignore[override]
        self.state = np.array([random.choice([0.0, 1.0])])
        self.steps = 0
        return self.state, {}  # Return the first observation


def test_mock_env_all_right() -> None:
    """Manually check the behavior of the mock environment. Perform all actions correctly."""
    max_steps = 10
    env = MockEnv(max_steps=max_steps)
    state, _ = env.reset()
    for _ in range(max_steps - 1):
        next_state, reward, done, _, _ = env.step(Action(int(state[0])))
        assert reward == 1
        assert not done
        state = next_state
    next_state, reward, done, _, _ = env.step(Action(int(state[0])))
    assert reward == 1
    assert done


def test_mock_env_all_wrong() -> None:
    """Manually check the behavior of the mock environment. Perform all actions incorrectly."""
    max_steps = 10
    env = MockEnv(max_steps=max_steps)
    state, _ = env.reset()
    for _ in range(max_steps - 1):
        next_state, reward, done, _, _ = env.step(Action(1 - int(state[0])))
        assert reward == -1
        assert not done
        state = next_state
    next_state, reward, done, _, _ = env.step(Action(1 - int(state[0])))
    assert reward == -1
    assert done


if __name__ == "__main__":
    test_mock_env_all_right()
    print("test_mock_env_all_right passed")
    test_mock_env_all_wrong()
    print("test_mock_env_all_wrong passed")


# %%
def test_reinforce_train() -> None:
    """Test the reinforce training loop on the mock environment."""
    env = MockEnv(max_steps=10)
    policy = Policy(state_size=1, action_size=2, hidden_sizes=[16])
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    scores = reinforce_train(
        env=env,
        policy=policy,
        optimizer=optimizer,
        gamma=1.0,
        num_episodes=100,
        max_t=10,
    )
    assert all(
        [score == 10 for score in scores[90:]]
    ), "The last 10 scores should be 10"


if __name__ == "__main__":
    test_reinforce_train()
    print("test_reinforce_train passed")

# %% [markdown]
# ## Real Environment
#
# Now that this has passed, we can be confident nothing obvious is wrong with the code, and we can move on to the cartpole environment.
#
# We choose a gamma that is non-1 to discount future rewards, but especially in the cartpole environment, setting it very close to 1 is beneficial, because the longer the pole is balanced, the more reward we get.
#
# We choose a small neural network and a fast learning rate, because this is not a hard problem.
#
# However, we do need to train longer than the hugging face tutorial, and I'm unsure why.

# %%
import plotly.express as px
from continuing_education.lib.experiments import ExperimentManager

if __name__ == "__main__":
    LR = 1e-2
    GAMMA = 1.0  # Cartpole benefits from a high gamma because the longer the pole is up, the higher the reward
    HIDDEN_SIZES = [16, 16]
    NUM_EPISODES = 10000
    MAX_T = 100
    # Do this a few times to prove consistency
    last_10_percent_mean = []
    for _ in range(3):
        env = gym.make("CartPole-v1")
        policy = Policy(
            state_size=OBSERVATION_SPACE_SHAPE[0],
            action_size=ACTION_SPACE_SIZE,
            hidden_sizes=HIDDEN_SIZES,
        ).to(DEVICE)
        optimizer = optim.Adam(policy.parameters(), lr=LR)
        scores = reinforce_train(
            env=env,
            policy=policy,
            optimizer=optimizer,
            gamma=GAMMA,
            num_episodes=NUM_EPISODES,
            max_t=MAX_T,
        )
        # Calculate the mean of the last 10 % of the scores
        last_10_percent_mean.append(
            sum(scores[int(NUM_EPISODES * 0.9) :]) / (NUM_EPISODES * 0.1)
        )
        fig = px.line(scores, title="Scores over time")
        fig.show()
    ExperimentManager(
        name="REINFORCE",
        description="Main Results",
        primary_metric="last_10_percent_mean",
        file=__this_file,
    ).commit(metrics={"last_10_percent_mean": last_10_percent_mean})


# %% [markdown]
# # Improvements
#
# 1. We should sample from actions instead of taking the argmax. This should lead to better exploration which reduces as the model gets more confident.
# 2. We should take a batch of episodes and optimize on the batch. This should lead to more stability in learning.


# %%
class SamplePolicy(Policy):
    """A classic policy network is one which takes in a state
    and returns a probability distribution over the action space.
    Act samples from the distribution instead of taking the greedy argmax."""

    def act(self, state: State, *, temperature: float) -> Tuple[Action, LogProb]:
        """Same as forward, instead of returning the entire distribution, we
        sample from the distribution
        along with the log probability of that action.
        In testing mode, you can set argmax=True to take the greedy action.
        """
        # First we got to convert out of numpy and into pytorch
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)

        # Now we can run the forward pass, whos output is a probability distribution
        # along the action space
        pdf = softmax_with_temperature(
            self.forward(state_tensor), temperature=temperature
        )
        assert torch.isclose(
            pdf.sum().cpu(), torch.Tensor([1.0])
        ).all(), "The output of the network should be a probability distribution"
        assert (
            pdf.cpu().shape[-1] == self.action_size
        ), "The output of the network should be a probability distribution over the action space"

        # Now we want to get the action that corresponds to the highest probability
        # TODO: We could sample from the pdf instead of taking the greedy argmax
        m = torch.distributions.Categorical(pdf)
        action_idx = m.sample()

        # We also need the log probability of the action
        # However, we are going to do backprop through the log probability of the action
        # Therefore this needs to stay as a tensor
        # The Category distribution in torch has a method for a backprop friendly log probability of one action from a multinomial distribution
        log_prob = m.log_prob(action_idx)

        # We return the action and the log probability of the action
        return Action(action_idx.item()), log_prob


# %%
def reinforce_train_batch(
    *,
    env: Env,
    policy: Policy,
    optimizer: optim.Optimizer,
    gamma: float,
    num_episodes: int,
    batch_size: int,
    max_t: int,
    temperature: float,
) -> list[float]:
    """Algorithm 1 REINFORCE modified to use batched episodes.
    equivalent to reinforce_train if batch_size=1
    """
    assert gamma <= 1, "Gamma should be less than or equal to 1"
    assert gamma > 0, "Gamma should be greater than 0"
    assert num_episodes > 0, "Number of episodes should be greater than 0"
    scores = []
    for _ in trange(num_episodes):
        policy_losses = []
        _scores = []
        for _ in range(batch_size):
            trajectory = collect_episode(
                env=env, policy=policy, max_t=max_t, temperature=temperature
            )
            _scores.append(sum([sar.reward for sar in trajectory]))
            policy_losses.append(
                objective(policy=policy, trajectory=trajectory, gamma=gamma)
            )
        policy_loss = torch.stack(cast(list[Tensor], policy_losses)).mean()
        scores.append(sum(_scores) / batch_size)
        optimizer.zero_grad()
        policy_loss.backward()  # This gives us the gradient
        optimizer.step()
    return scores


def test_reinforce_train_batch() -> None:
    """Test the reinforce training loop on the mock environment."""
    env = MockEnv(max_steps=10)
    policy = SamplePolicy(state_size=1, action_size=2, hidden_sizes=[16])
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    scores = reinforce_train_batch(
        env=env,
        policy=policy,
        optimizer=optimizer,
        gamma=1.0,
        num_episodes=100,
        batch_size=10,
        max_t=10,
        temperature=1.0,
    )
    assert all(
        [score >= 9 for score in scores[90:]]
    ), f"The last 10 scores should be 10, maybe some 9s. Got: {scores}"


if __name__ == "__main__":
    test_reinforce_train_batch()
    print("test_reinforce_train_batch passed")

# %%
if __name__ == "__main__":
    BATCH_SIZE = 10
    NUM_EPISODES = 200
    # Do this a few times to prove consistency
    last_10_percent_mean = []
    for _ in range(3):
        env = gym.make("CartPole-v1")
        policy = SamplePolicy(
            state_size=OBSERVATION_SPACE_SHAPE[0],
            action_size=ACTION_SPACE_SIZE,
            hidden_sizes=HIDDEN_SIZES,
        ).to(DEVICE)
        optimizer = optim.Adam(policy.parameters(), lr=LR)
        scores = reinforce_train_batch(
            env=env,
            policy=policy,
            optimizer=optimizer,
            gamma=GAMMA,
            num_episodes=NUM_EPISODES,
            batch_size=BATCH_SIZE,
            max_t=MAX_T,
            temperature=1.0,
        )
        # Calculate the mean of the last 10 % of the scores
        last_10_percent_mean.append(
            sum(scores[int(NUM_EPISODES * 0.9) :]) / (NUM_EPISODES * 0.1)
        )
        fig = px.line(scores, title="Scores over time")
        fig.show()
    ExperimentManager(
        name="REINFORCE",
        description=f"Batch Size {BATCH_SIZE} + Sample Results",
        primary_metric="last_10_percent_mean",
        file=__this_file,
    ).commit(metrics={"last_10_percent_mean": last_10_percent_mean})

# %% [markdown]
# This worked well! Now I'm curious to see which improvement independently mattered the most, or if both worked together to improve the model.

# %% [markdown]
# ### Argmax, with batching

# %%
if __name__ == "__main__":
    BATCH_SIZE = 10
    # Do this a few times to prove consistency
    last_10_percent_mean = []
    for _ in range(3):
        env = gym.make("CartPole-v1")
        policy = Policy(
            state_size=OBSERVATION_SPACE_SHAPE[0],
            action_size=ACTION_SPACE_SIZE,
            hidden_sizes=HIDDEN_SIZES,
        ).to(DEVICE)
        optimizer = optim.Adam(policy.parameters(), lr=LR)
        scores = reinforce_train_batch(
            env=env,
            policy=policy,
            optimizer=optimizer,
            gamma=GAMMA,
            num_episodes=NUM_EPISODES,
            batch_size=BATCH_SIZE,
            max_t=MAX_T,
            temperature=1.0,
        )
        # Calculate the mean of the last 10 % of the scores
        last_10_percent_mean.append(
            sum(scores[int(NUM_EPISODES * 0.9) :]) / (NUM_EPISODES * 0.1)
        )
        fig = px.line(scores, title="Scores over time")
        fig.show()
    ExperimentManager(
        name="REINFORCE",
        description=f"Batch Size {BATCH_SIZE} + Argmax Results",
        primary_metric="last_10_percent_mean",
        file=__this_file,
    ).commit(metrics={"last_10_percent_mean": last_10_percent_mean})

# %% [markdown]
# ### Sampling, no batching

# %%
if __name__ == "__main__":
    BATCH_SIZE = 1
    # Do this a few times to prove consistency
    last_10_percent_mean = []
    for _ in range(3):
        env = gym.make("CartPole-v1")
        policy = SamplePolicy(
            state_size=OBSERVATION_SPACE_SHAPE[0],
            action_size=ACTION_SPACE_SIZE,
            hidden_sizes=HIDDEN_SIZES,
        ).to(DEVICE)
        optimizer = optim.Adam(policy.parameters(), lr=LR)
        scores = reinforce_train(
            env=env,
            policy=policy,
            optimizer=optimizer,
            gamma=GAMMA,
            num_episodes=NUM_EPISODES,
            max_t=MAX_T,
        )
        # Calculate the mean of the last 10 % of the scores
        last_10_percent_mean.append(
            sum(scores[int(NUM_EPISODES * 0.9) :]) / (NUM_EPISODES * 0.1)
        )
        fig = px.line(scores, title="Scores over time")
        fig.show()
    ExperimentManager(
        name="REINFORCE",
        description=f"Batch Size {BATCH_SIZE} + Sample Results",
        primary_metric="last_10_percent_mean",
        file=__this_file,
    ).commit(metrics={"last_10_percent_mean": last_10_percent_mean})

# %% [markdown]
# ### Both, but with smaller batch size

# %%
if __name__ == "__main__":
    BATCH_SIZE = 2
    # Do this a few times to prove consistency
    last_10_percent_mean = []
    for _ in range(3):
        env = gym.make("CartPole-v1")
        policy = SamplePolicy(
            state_size=OBSERVATION_SPACE_SHAPE[0],
            action_size=ACTION_SPACE_SIZE,
            hidden_sizes=HIDDEN_SIZES,
        ).to(DEVICE)
        optimizer = optim.Adam(policy.parameters(), lr=LR)
        scores = reinforce_train_batch(
            env=env,
            policy=policy,
            optimizer=optimizer,
            gamma=GAMMA,
            num_episodes=NUM_EPISODES,
            batch_size=BATCH_SIZE,
            max_t=MAX_T,
            temperature=1.0,
        )
        # Calculate the mean of the last 10 % of the scores
        last_10_percent_mean.append(
            sum(scores[int(NUM_EPISODES * 0.9) :]) / (NUM_EPISODES * 0.1)
        )
        fig = px.line(scores, title="Scores over time")
        fig.show()
    ExperimentManager(
        name="REINFORCE",
        description=f"Batch Size {BATCH_SIZE} + Sample Results",
        primary_metric="last_10_percent_mean",
        file=__this_file,
    ).commit(metrics={"last_10_percent_mean": last_10_percent_mean})

# %% [markdown]
# ## Conclusion
#
# REINFORCE seems to be rather unstable without some degree of batching, and without some degree of forced exploration, in this case done via sampling from the output distribution. It reliably learns the cartpole environment with a BATCH_SIZE of 10 and a sampling strategy. Doing one or the other or neither of Batching and sampling leads to very noisy results, which can lead to catestrophic forgetting. We will explore other strategies and algorithms in future notebooks.

# %% [markdown]
# # References
#
# 1. Williams, R.J. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Mach Learn 8, 229–256 (1992). https://doi.org/10.1007/BF00992696
#
# 2. UNIT 4. POLICY GRADIENT WITH PYTORCH. Hugging Face. (n.d.). https://huggingface.co/learn/deep-rl-course/unit4

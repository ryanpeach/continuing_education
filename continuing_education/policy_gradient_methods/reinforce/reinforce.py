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

def get_environment_space(env_name: str) -> tuple[tuple[int, ...], int]:
    env = gym.make(env_name)
    observation_space_shape = env.observation_space.shape
    action_space_size = env.action_space.n
    print("State size:", observation_space_shape)
    print("Action size:", action_space_size)
    state = env.reset()
    print(f"Example state: {state}")
    action_return = env.step(1)
    print(f"Action return: {action_return}")
    return observation_space_shape, action_space_size

if __name__=="__main__":
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
State = NewType("State", npt.NDArray[np.float64])
Action = NewType("Action", int)
Reward = NewType("Reward", float)
LogProb = NewType("LogProb", Tensor)
Loss = NewType("Loss", Tensor)

# %%
from typing import List, Tuple
from torch import nn


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
        network = []
        network.append(nn.Linear(state_size, hidden_sizes[0]))  # Shape: (:, state_size, hidden_sizes[0])
        network.append(nn.ReLU())
        for i in range(len(hidden_sizes) - 1):
            network.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))  # Shape: (:, hidden_sizes[i], hidden_sizes[i+1])
            network.append(nn.ReLU())
        network.append(nn.Linear(hidden_sizes[-1], action_size))  # Shape: (:, hidden_sizes[-1], action_size)
        network.append(nn.Softmax(dim=-1))  # Softmax along the action dimension
        self.network = nn.Sequential(*network).to(DEVICE)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Takes a state tensor and returns a probability distribution along the action space"""
        state = state.to(DEVICE)
        return self.network(state)
    
    def act(self, state: State) -> Tuple[Action, LogProb]:
        """Same as forward, instead of returning the entire distribution, we
        return the maximum probability action
        along with the log probability of that action
        """
        # First we got to convert out of numpy and into pytorch
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)

        # Now we can run the forward pass, whos output is a probability distribution
        # along the action space
        pdf = self.forward(state_tensor)
        assert torch.isclose(pdf.sum().cpu(), torch.Tensor([1.0])).all(), "The output of the network should be a probability distribution"
        assert pdf.cpu().shape[-1] == self.action_size, "The output of the network should be a probability distribution over the action space"

        # Now we want to get the action that corresponds to the highest probability
        # TODO: We could sample from the pdf instead of taking the greedy argmax
        action_idx = torch.argmax(pdf)

        # We also need the log probability of the action
        # However, we are going to do backprop through the log probability of the action
        # Therefore this needs to stay as a tensor
        # The Category distribution in torch has a method for a backprop friendly log probability of one action from a multinomial distribution
        log_prob = torch.distributions.Categorical(pdf).log_prob(action_idx)

        # We return the action and the log probability of the action
        return Action(action_idx.item()), log_prob


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
def normalize(returns: Tensor) -> Tensor:
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

def collect_episode(*, env: Env, policy: Policy, max_t: int = 1000) -> Trajectory:
    """2.1 Returns the trajectory of one episode of using the policy.
    
    The output is a list of SAR tuples, where each tuple represents a state, action, reward tuple.

    In the hugging face tutorial this is represented as:
    
    $S_0, A_0, r_0, ..., S_{T-1}, A_{T-1}, r_{T-1}$
    """
    state, _ = env.reset()
    done = False
    trajectory = []
    for _ in range(max_t):
        action, log_prob = policy.act(state)
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
CumDiscFutureRewardTrajectory = NewType("CumDiscFutureRewardTrajectory", RewardTrajectory)

def cumulative_discounted_future_rewards(
    *, trajectory: RewardTrajectory, gamma: float
) -> CumDiscFutureRewardTrajectory:
    """2.2.1 Returns the cumulative discounted future rewards of a trajectory at each step of the trajectory.
    
    In the hugging face tutorial,
    each element in the output is represented by $G_t$ where $t$ is the index of the element."""
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


def test_cumulative_discounted_future_rewards() -> None:
    # It's important to test our code, so we know it works as expected
    # We tried to use ipytest but it wasn't working https://github.com/chmp/ipytest
    assert cumulative_discounted_future_rewards(
        trajectory=RewardTrajectory([-1]), gamma=0.5
    ) == RewardTrajectory([-1])
    assert cumulative_discounted_future_rewards(
        trajectory=RewardTrajectory([0]), gamma=0.5
    ) == RewardTrajectory([0])
    assert cumulative_discounted_future_rewards(
        trajectory=RewardTrajectory([0, 1]), gamma=0.5
    ) == RewardTrajectory([0.5, 1])
    assert cumulative_discounted_future_rewards(
        trajectory=RewardTrajectory([0, 1, 1]), gamma=0.5
    ) == RewardTrajectory([0.75, 1.5, 1])


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
        cumulative_discounted_future_rewards(
            trajectory=RewardTrajectory([sar.reward for sar in trajectory]), gamma=gamma
        )
    )
    for cum_disc_reward, sar in zip(cum_disc_rewards, trajectory):
        _, action_log_prob = policy.act(sar.state)
        loss.append(cum_disc_reward * -action_log_prob)  # This is negative to turn maximization into minimization
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

def reinforce_train(*, env: Env, policy: Policy, optimizer: optim.Optimizer, gamma: float, num_episodes: int):
    """Algorithm 1 REINFORCE"""
    assert gamma <= 1, "Gamma should be less than or equal to 1"
    assert gamma > 0, "Gamma should be greater than 0"
    assert num_episodes > 0, "Number of episodes should be greater than 0"
    scores = []
    for _ in trange(num_episodes):
        # TODO: We could batch these episodes to get more stability
        trajectory = collect_episode(env=env, policy=policy)
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
import gym
from gym import spaces
import random

from numpy import float32

class MockEnv(gym.Env):
    """A dead simple environment for reinforcement learning that rewards the agent for going left.
    Useful for unit testing.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, max_steps=10):
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(1)
        self.max_steps = max_steps
        self.reset()

    def step(self, action: Action) -> tuple[npt.NDArray[float32], Reward, bool, bool, dict]:
        assert self.action_space.contains(action), f"Invalid Action: {action}"
        
        # If the state is 0, then the 0th action is the correct action
        # If the state is 1, then the 1st action is the correct action
        reward = 1 if action == self.state else -1
        self.steps += 1
        done = self.steps >= self.max_steps
        if done:
            return self.state, reward, done, False, {}
        else:
            self.state = np.array([random.choice([0.0, 1.0])])
            return self.state, reward, done, False, {}
        
    def reset(self) -> tuple[npt.NDArray[float32], dict]:
        self.state = np.array([random.choice([0.0, 1.0])])
        self.steps = 0
        return self.state, {}  # Return the first observation

def test_mock_env_all_right() -> None:
    """Manually check the behavior of the mock environment. Perform all actions correctly."""
    max_steps = 10
    env = MockEnv(max_steps=max_steps)
    state, _ = env.reset()
    for _ in range(max_steps-1):
        next_state, reward, done, _, _ = env.step(int(state[0]))
        assert reward == 1
        assert done == False
        state = next_state
    next_state, reward, done, _, _ = env.step(int(state[0]))
    assert reward == 1
    assert done == True

def test_mock_env_all_wrong() -> None:
    """Manually check the behavior of the mock environment. Perform all actions incorrectly."""
    max_steps = 10
    env = MockEnv(max_steps=max_steps)
    state, _ = env.reset()
    for _ in range(max_steps-1):
        next_state, reward, done, _, _ = env.step(1-int(state[0]))
        assert reward == -1
        assert done == False
        state = next_state
    next_state, reward, done, _, _ = env.step(1-int(state[0]))
    assert reward == -1
    assert done == True

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
    scores = reinforce_train(env=env, policy=policy, optimizer=optimizer, gamma=1.0, num_episodes=100)
    assert all([score == 10 for score in scores[90:]]), "The last 10 scores should be 10"

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
if __name__ == "__main__":
    LR = 1e-2
    GAMMA = 1.0 # Cartpole benefits from a high gamma because the longer the pole is up, the higher the reward
    HIDDEN_SIZES = [16, 16]
    NUM_EPISODES= 1000
    env = gym.make("CartPole-v1")
    policy = Policy(
        state_size=OBSERVATION_SPACE_SHAPE[0],
        action_size=ACTION_SPACE_SIZE,
        hidden_sizes=HIDDEN_SIZES,
    ).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    scores = reinforce_train(env=env, policy=policy, optimizer=optimizer, gamma=GAMMA, num_episodes=NUM_EPISODES)

# %% [markdown]
# ## Results
#
# It's pretty easy to tell if we succeeded or not. If the scores over time increase to the ceiling of `max_t` and stay there consistently, we have succeeded. If they do not, we have failed.

# %%
import plotly.express as px

if __name__=="__main__":
    fig = px.line(scores, title="Scores over time")
    fig.show()


# %% [markdown]
# # Improvements
#
# 1. We should sample from actions instead of taking the argmax
# 2. We should take a batch of episodes and optimize on the batch
#
# These should minimize the instability of the training process seen in the above graph, where it sometimes looses the gains it has made.

# %%
class SamplePolicy(Policy):
    """A classic policy network is one which takes in a state
    and returns a probability distribution over the action space.
    Act samples from the distribution instead of taking the greedy argmax."""

    def act(self, state: State) -> Tuple[Action, LogProb]:
        """Same as forward, instead of returning the entire distribution, we
        sample from the distribution
        along with the log probability of that action.
        In testing mode, you can set argmax=True to take the greedy action.
        """
        # First we got to convert out of numpy and into pytorch
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)

        # Now we can run the forward pass, whos output is a probability distribution
        # along the action space
        pdf = self.forward(state_tensor)
        assert torch.isclose(pdf.sum().cpu(), torch.Tensor([1.0])).all(), "The output of the network should be a probability distribution"
        assert pdf.cpu().shape[-1] == self.action_size, "The output of the network should be a probability distribution over the action space"

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
def reinforce_train_batch(*, env: Env, policy: Policy, optimizer: optim.Optimizer, gamma: float, num_episodes: int, batch_size: int) -> None:
    """Algorithm 1 REINFORCE"""
    assert gamma <= 1, "Gamma should be less than or equal to 1"
    assert gamma > 0, "Gamma should be greater than 0"
    assert num_episodes > 0, "Number of episodes should be greater than 0"
    scores = []
    for _ in trange(num_episodes):
        policy_losses = []
        _scores = []
        for _ in range(batch_size):
            trajectory = collect_episode(env=env, policy=policy)
            _scores.append(sum([sar.reward for sar in trajectory]))
            policy_losses.append(objective(policy=policy, trajectory=trajectory, gamma=gamma))
        policy_loss = torch.stack(policy_losses).mean()
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
    scores = reinforce_train_batch(env=env, policy=policy, optimizer=optimizer, gamma=1.0, num_episodes=100, batch_size=10)
    assert all([score >= 9 for score in scores[90:]]), f"The last 10 scores should be 10, maybe some 9s. Got: {scores}"

if __name__ == "__main__":
    test_reinforce_train_batch()
    print("test_reinforce_train_batch passed")

# %%
if __name__ == "__main__":
    BATCH_SIZE = 10
    env = gym.make("CartPole-v1")
    policy = SamplePolicy(
        state_size=OBSERVATION_SPACE_SHAPE[0],
        action_size=ACTION_SPACE_SIZE,
        hidden_sizes=HIDDEN_SIZES,
    ).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    scores = reinforce_train_batch(env=env, policy=policy, optimizer=optimizer, gamma=GAMMA, num_episodes=NUM_EPISODES, batch_size=BATCH_SIZE)
    fig = px.line(scores, title="Scores over time")
    fig.show()

# %% [markdown]
# This is interesting. These changes actually seem to have hurt the training process. Lets see if its argmax or the batching that is the problem.
#
# ### Argmax, with batching

# %%
if __name__ == "__main__":
    BATCH_SIZE = 10
    env = gym.make("CartPole-v1")
    policy = Policy(
        state_size=OBSERVATION_SPACE_SHAPE[0],
        action_size=ACTION_SPACE_SIZE,
        hidden_sizes=HIDDEN_SIZES,
    ).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    scores = reinforce_train_batch(env=env, policy=policy, optimizer=optimizer, gamma=GAMMA, num_episodes=NUM_EPISODES, batch_size=BATCH_SIZE)
    fig = px.line(scores, title="Scores over time")
    fig.show()

# %% [markdown]
# ## Sampling, no batching

# %%
if __name__ == "__main__":
    BATCH_SIZE = 10
    env = gym.make("CartPole-v1")
    policy = SamplePolicy(
        state_size=OBSERVATION_SPACE_SHAPE[0],
        action_size=ACTION_SPACE_SIZE,
        hidden_sizes=HIDDEN_SIZES,
    ).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    scores = reinforce_train(env=env, policy=policy, optimizer=optimizer, gamma=GAMMA, num_episodes=NUM_EPISODES)
    fig = px.line(scores, title="Scores over time")
    fig.show()

# %% [markdown]
# ## Conclusion TODO

# %% [markdown]
# # References
#
# 1. Williams, R.J. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Mach Learn 8, 229–256 (1992). https://doi.org/10.1007/BF00992696
#
# 2. UNIT 4. POLICY GRADIENT WITH PYTORCH. Hugging Face. (n.d.). https://huggingface.co/learn/deep-rl-course/unit4

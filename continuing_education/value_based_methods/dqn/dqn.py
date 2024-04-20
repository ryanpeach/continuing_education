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
#     display_name: continuing_education
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from pathlib import Path

if __name__ == "__main__":
    __this_file = Path().resolve() / "dqn.ipynb"  # jupyter does not have __file__

# %%
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    print(DEVICE)

# %%
from torch import nn

from continuing_education.policy_gradient_methods.reinforce import collect_episode, SamplePolicy, Action, State, Env, cumulative_discounted_future_rewards

# %% [markdown]
# # Q Learning
#
# Lets quickly create a simple Q Learning Agent and test it on cartpole environment.

# %%

from typing import Any

class QLearningModel(nn.Module):
    def __init__(
        self, *, state_size: int, action_size: int, hidden_sizes: list[int]
    ) -> None:
        """
        Notice that this is exactly the same as the Policy network from REINFORCE, because
        we are still starting from the state and outputting an action. The difference is that
        we will not softmax the output, because its not a probability distribution, but rather
        a regressor that outputs the Q value of each action.
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

    def act(self, state: State) -> Action:
        """
        Same as the policy network, but instead of softmaxing and sampling, 
        the network actually is a regressor returning real numbered values, and we are argmaxing over them.
        We don't get a log_prob, and we don't pass a temperature, because Q networks cant handle stochastic policies.
        """
        # First we got to convert out of numpy and into pytorch
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)

        # Now we can run the forward pass, whos output is a probability distribution
        # along the action space
        action_values = self.forward(state_tensor)
        assert (
            action_values.cpu().shape[-1] == self.action_size
        ), "The output of the network should be a probability distribution over the action space"

        # Now we want to get the action that corresponds to the highest probability
        # TODO: We could sample from the pdf instead of taking the greedy argmax
        action_idx = torch.argmax(action_values)

        # We return the action and the log probability of the action
        return Action(int(action_idx.item()))


# %%
from dataclasses import dataclass
import random
from collections import deque
from typing import NewType

@dataclass
class SARS:
    state: State
    action: Action
    reward: float
    next_state: State
    done: bool

Trajectory = NewType("Trajectory", list[SARS])

# %%
from typing import Generator
from gym import Env

from continuing_education.policy_gradient_methods.reinforce.reinforce import Reward


def collect_episode(
    *, env: Env, value_network: QLearningModel, max_t: int = 1000
) -> Generator[Trajectory, None, None]:
    """2.1 Returns the trajectory of one episode of using the value network.

    The output is a list of SARS tuples, where each tuple represents a state, action, reward, next_state tuple.
    """
    state, _ = env.reset()
    done = False
    for _ in range(max_t):
        action = value_network.act(state)
        next_state, reward, done, _, _ = env.step(action)
        yield SARS(
            state=State(state),
            action=action,
            reward=Reward(reward),
            next_state=State(next_state),
            done=done,
        )
        state = next_state
        if done:
            break



# %%

class ActionReplayMemory:
    """The simplest kind of memory buffer for q learning.
    This is a FIFO buffer of a fixed length that stores SAR objects from `continuing_education.policy_gradient_methods.reinforce.collect_episode`.
    These SAR objects have been modified already using `continuing_education.policy_gradient_methods.reinforce.cumulative_discounted_future_rewards`
    to replace their reward values.
    """
    def __init__(self, max_size: int) -> None:
        self.buffer: deque[SARS] = deque(maxlen=max_size)

    def push(self, item: SARS) -> None:
        self.buffer.append(item)

    def sample(self, batch_size: int) -> list[SARS]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


# %% [markdown]
# # Train

# %%
from torch import Tensor
from tqdm.notebook import trange
import torch.optim as optim

def objective(
    *,
    value_network: QLearningModel,
    batch: list[SARS],
    gamma: float,
) -> Tensor:
    """The objective function for the DQN algorithm is simple regression loss."""
    states = torch.tensor([s.state for s in batch]).float().to(DEVICE) # shape (batch_size, state_size)
    assert states.shape[1] == value_network.state_size, "The state size of the value network should match the state size of the batch"
    rewards = torch.tensor([s.reward for s in batch]).float().to(DEVICE).unsqueeze(1) # shape (batch_size, 1)
    assert rewards.shape[1] == 1, "The reward should be a scalar value"
    next_states = torch.tensor([s.next_state for s in batch]).float().to(DEVICE) # shape (batch_size, state_size)
    assert next_states.shape[1] == value_network.state_size, "The state size of the value network should match the state size of the batch"
    dones = torch.tensor([s.done for s in batch]).float().to(DEVICE).unsqueeze(1) # shape (batch_size, 1)
    assert dones.shape[1] == 1, "The done should be a scalar value"

    # We are going to use the value network to predict the Q values for the current state
    predicted_q_values = value_network.forward(states) # shape (batch_size, action_size)
    assert predicted_q_values.shape[0] == len(batch), "The first dimension of the output should match the batch size"
    assert predicted_q_values.shape[1] == value_network.action_size, "The output of the network should be the same size as the action space"
    
    # We are going to use the value network to predict the Q values for the next state
    next_predicted_q_values = value_network.forward(next_states) # shape (batch_size, action_size)
    assert next_predicted_q_values.shape[0] == len(batch), "The first dimension of the output should match the batch size"
    assert next_predicted_q_values.shape[1] == value_network.action_size, "The output of the network should be the same size as the action space"

    # Generate the Q Loss using the bellman equation
    # Q(s, a) = r + gamma * max_a'(Q(s', a'))
    next_action_value_predicted = next_predicted_q_values.max(1).values.unsqueeze(1)
    bellman = rewards + gamma * next_action_value_predicted * (1.0 - dones)
    assert bellman.shape[0] == len(batch), "The first dimension of the output should match the batch size"
    assert bellman.shape[1] == 1, "The bellman equation should output a scalar value"

    loss = nn.MSELoss()(predicted_q_values.max(1).values.unsqueeze(1), bellman)
    assert loss.shape == (), "The loss should be a scalar value"

    return loss


# %%
def dqn_train(
    *,
    env: Env,
    value_network: QLearningModel,
    memory: ActionReplayMemory,
    optimizer: optim.Optimizer,
    gamma: float,
    num_episodes: int,
    max_t: int,
    batch_size: int,
) -> list[Reward]:
    """Algorithm 1 REINFORCE"""
    assert gamma <= 1, "Gamma should be less than or equal to 1"
    assert gamma > 0, "Gamma should be greater than 0"
    assert num_episodes > 0, "Number of episodes should be greater than 0"
    scores: list[Reward] = []
    for _ in trange(num_episodes):
        _scores = []
        for sars in collect_episode(
            env=env, value_network=value_network, max_t=max_t
        ):
            memory.push(sars)
            _scores.append(sars.reward)
            if len(memory) > batch_size:
                batch = memory.sample(batch_size)
                loss = objective(value_network=value_network, batch=batch, gamma=gamma)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        scores.append(sum(_scores))

    return scores


# %%
from continuing_education.policy_gradient_methods.reinforce.reinforce import get_environment_space


if __name__ == "__main__":
    OBSERVATION_SPACE_SHAPE, ACTION_SPACE_SIZE = get_environment_space("CartPole-v1")

# %%
import gym
import plotly.express as px
from continuing_education.lib.experiments import ExperimentManager

if __name__ == "__main__":
    LR = 1e-2
    GAMMA = 1.0  # Cartpole benefits from a high gamma because the longer the pole is up, the higher the reward
    HIDDEN_SIZES = [16, 16]
    NUM_EPISODES = 1000
    MAX_T = 100
    BATCH_SIZE = 64
    MAX_MEMORY = 10000
    # Do this a few times to prove consistency
    last_10_percent_mean = []
    for _ in range(3):
        env = gym.make("CartPole-v1")
        value_network = QLearningModel(
            state_size=OBSERVATION_SPACE_SHAPE[0],
            action_size=ACTION_SPACE_SIZE,
            hidden_sizes=HIDDEN_SIZES,
        ).to(DEVICE)
        optimizer = optim.Adam(value_network.parameters(), lr=LR)
        scores = dqn_train(
            env=env,
            value_network=value_network,
            optimizer=optimizer,
            gamma=GAMMA,
            num_episodes=NUM_EPISODES,
            max_t=MAX_T,
            memory=ActionReplayMemory(MAX_MEMORY),
            batch_size=64,
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

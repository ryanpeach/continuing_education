from continuing_education.lib.types import DiscreteAction as Action, LogProb, State


import torch
from torch import nn


from abc import ABC, abstractmethod


class DiscreteActionPolicyInterface(nn.Module, ABC):
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

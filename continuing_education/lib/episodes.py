from dataclasses import dataclass

from gym import Env
from torch import tensor
from continuing_education.lib.interfaces import DiscreteActionPolicyInterface
from continuing_education.lib.types import DiscreteAction as Action
from continuing_education.lib.types import State, Done, LogProb, Reward

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
    action_log_prob: LogProb = LogProb(tensor(0.0))

    def to_sars(self) -> "SARS":
        return SARS.from_sarsa(self)


def collect_episode(
    *, env: Env, policy: DiscreteActionPolicyInterface, max_t: int, **policy_kwargs
) -> Generator[SARSA, None, None]:
    """A generator that yields SARSA tuples for a single episode."""
    state, _ = env.reset()
    action, action_logprob = policy.act(state, **policy_kwargs)

    for _ in range(max_t):
        next_state, reward, done, _, _ = env.step(action)
        next_action: Action | None = None
        next_action_logprob = LogProb(tensor(0.0))
        if not done:
            next_action, next_action_logprob = policy.act(next_state, **policy_kwargs)

        yield SARSA(
            state=State(state),
            action=Action(action),
            reward=Reward(reward),
            next_state=State(next_state),
            next_action=Action(next_action) if next_action is not None else None,
            done=Done(done),
            action_log_prob=LogProb(action_logprob),
        )

        if done:
            break

        assert next_action is not None
        state, action, action_logprob = next_state, next_action, next_action_logprob


@dataclass
class SARS:
    state: State
    action: Action
    reward: float
    next_state: State
    done: bool

    @staticmethod
    def from_sarsa(sarsa: SARSA) -> "SARS":
        return SARS(
            state=sarsa.state,
            action=sarsa.action,
            reward=sarsa.reward,
            next_state=sarsa.next_state,
            done=sarsa.done,
        )

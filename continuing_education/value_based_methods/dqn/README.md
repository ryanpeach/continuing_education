# DQN

## What is the bellman equation?
* The Bellman equation is a recursive relationship that defines the [[value]] of a state-action pair in terms of the immediate reward and the expected value of the next state.

* $Q(s, a) = r + \lambda * max_a'(Q(s', a')) * (1 - done(s, a))$
  * $Q(s, a)$: The Q-value of taking action $a$ in state $s$
  * $r$: The reward of taking action $a$ in state $s$
  * $\lambda$: The discount factor
  * $max_a'(Q(s', a'))$: The maximum Q-value of the next state $s'$
  * $done(s, a)$: Whether the episode is done after taking action $a$ in state $s$

## What does the Q [[value]] represent?

The expected cumulative future reward of taking action $a$ in state $s$ following the optimal [[policy]] thereafter.

## What is the objective function of DQN?

* $L(\theta) = \mathbb{E}[(Q_\theta(s, a) - (r + \lambda * max_a'(Q_\theta(s', a'))))^2]$
  * $L(\theta)$: The loss function
  * $Q(s, a)$: The Q-value of taking action $a$ in state $s$
  * $Q(s', a')$: The Q-value of taking action $a'$ in next state $s'$
  * $r$: The reward of taking action $a$ in state $s$
  * $\lambda$: The discount factor

## What are some differences in the `act` method between [[continuing_education/value_based_methods/dqn/README|dqn]] and [[continuing_education/policy_gradient_methods/reinforce/README|REINFORCE]]

* QLearning uses an argmax to select the best action whereas [[continuing_education/policy_gradient_methods/reinforce/README|REINFORCE]] uses a softmax sample to select an action
* QLearning has an epsilon greedy [[policy]] whereas [[continuing_education/policy_gradient_methods/reinforce/README|REINFORCE]] has a stochastic [[policy]]

## Define an epsilon greedy [[policy]]

* An epsilon greedy [[policy]] is a [[policy]] that selects the best action with probability $1 - \epsilon$ and a random action with probability $\epsilon$

## What are some differences in the `train` method between [[continuing_education/value_based_methods/dqn/README|dqn]] and [[continuing_education/policy_gradient_methods/reinforce/README|REINFORCE]]

* QLearning trains on each step whereas [[continuing_education/policy_gradient_methods/reinforce/README|REINFORCE]] trains at the end of each episode
* [[continuing_education/policy_gradient_methods/reinforce/README|REINFORCE]] needs a whole trajectory to train, because it operates on real cumulative rewards, whereas QLearning can train on each step because it operates on predicted cumulative rewards

## What are some differences in the `collect_episodes` method between [[continuing_education/value_based_methods/dqn/README|dqn]] and [[continuing_education/policy_gradient_methods/reinforce/README|REINFORCE]]

* QLearning uses SARS whereas [[continuing_education/policy_gradient_methods/reinforce/README|REINFORCE]] uses SAR
* [[continuing_education/policy_gradient_methods/reinforce/README|REINFORCE]] needs a whole trajectory to train, because it operates on real cumulative rewards, whereas QLearning can train on each step because it operates on predicted cumulative rewards
* This is because the bellman equation requires a mixture of one real reward and one predicted reward from the network to properly train

## What are the differences between exploration rate and temperature in [[continuing_education/value_based_methods/dqn/README|dqn]] and [[continuing_education/policy_gradient_methods/reinforce/README|REINFORCE]]?

* Exploration rate is a probability of taking a random action
* Temperature is a parameter in the softmax function that controls the stochasticity of the [[policy]]
* Exploration either happens or does not happen and is not controlled by the neural network at all
* Temperature just controls the output of the network. If the network is very confident in one action, it will still take that action with high probability, but if the network is unsure, it will take a random action with some probability. Therefore you don't need to decay temperature, but you do need to decay exploration rate.

## What is action replay memory? Why is it needed in [[continuing_education/value_based_methods/dqn/README|dqn]] but not [[continuing_education/policy_gradient_methods/reinforce/README|REINFORCE]].

* A buffer that stores experiences for training the network.
* [[continuing_education/value_based_methods/dqn/README|dqn]] is an off-policy method, so it can learn from past experiences. [[continuing_education/policy_gradient_methods/reinforce/README|REINFORCE]] is an on-policy method, so it can't learn from past experiences. The replay buffer is a feature of off-policy methods, not a hindrance.

## What is on-policy learning?

When the agent learns from the same [[policy]] that it uses to interact with the environment.

## What is off-policy learning?

Off-policy learning is when the agent can learn from a different [[policy]] than the one it uses to interact with the environment. This allows for more efficient learning because the agent can learn from past experiences.

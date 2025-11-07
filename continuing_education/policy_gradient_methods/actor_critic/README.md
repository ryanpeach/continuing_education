# Actor Critic

## What is the [[value]] function used for in [[continuing_education/value_based_methods/dqn/README|actor critic]] methods?
    - The [[value]] function is used to estimate the *average* expected return from a given state.
    - It could theoretically be a Q-function, but in practice, it is often a state-value function using [[TD]] error.
## What is the [[advantage]] function used for in [[continuing_education/value_based_methods/dqn/README|actor critic]] methods?
    - The difference between the expected reward from a state-action pair (Q) and the average expected reward from just the state (V).
        - $A(s, a) = Q(s, a) - V(s)$
    - It is used to normalize the [[continuing_education/policy_gradient_methods/README|policy-gradient]], as well as to push the [[continuing_education/policy_gradient_methods/README|policy-gradient]] towards actions that are better than average and away from actions that are worse than average.
## What is the training loop for a2c? #todo
## What is the difference between a2c and a3c?

## What is the [[value]] function used for in [[actor critic]] methods?

- The [[value]] function is used by the critic to evaluate the expected return from a given state or state-action pair.
- It provides a baseline for the [[policy gradient]] updates, helping to reduce variance in the learning process. It does this by providing an estimate of the expected return from a state, which can be subtracted from the actual return to compute the state [[advantage]] function.
- The [[value]] function is used to estimate the *average* expected return from a given state.
- It could theoretically be a Q-function, but in practice, it is often a [[value]] function.

source: Myself

## What is the [[advantage]] function used for in [[actor critic]] methods?

- The difference between the expected reward from a state-action pair (Q) and the average expected reward from just the state (V).
    - $A(s, a) = Q(s, a) - V(s)$
- It is normalized making the [[policy gradient]] normalized as well (providing updates around +-0)
- It pushes the [[policy gradient]] towards actions that are better than average and away from actions that are worse than average.

source: Myself

## What kinds of normalization are used in [[actor critic]] methods? What are their effects?



## What is the training loop for A2C?

## What is the training loop for A3c?

## What does A2C stand for?

A2C stands for Advantage Actor-Critic.

## What does A3C stand for?

A3C stands for Asynchronous Advantage Actor-Critic.

## What is the difference between A2C and A3C?

A3C is parallel and asynchronous, meaning it uses multiple agents to explore the environment in parallel and updates the model asynchronously. A2C is synchronous, meaning it uses a single agent to explore the environment and updates the model synchronously.

source: https://en.wikipedia.org/wiki/Actor-critic_algorithm#Variants

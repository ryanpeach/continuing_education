# Dueling DQN

## What is the dueling DQN architecture?

The Dueling [[dqn]] architecture is a modification of the standard [[dqn]] architecture that separates the [[value]] function and [[advantage]]  function into two separate streams.

## What is the mathematical identity linking the advantage function, the Q function, and the value function?

Because the [[advantage]] function can be defined as $A(s,a) = Q(s,a) - V(s)$, we can re-arrange this to get the Q-value function as $Q(s,a) = A(s,a) - V(s)$.

## What advantage does separating the advantage and value functions provide?

This has the effect of constraining the bias of each network, which can help with convergence.

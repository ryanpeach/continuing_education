---
alias: dueling deep q network
---

- The Dueling DQN architecture is a modification of the standard [[DQN]] architecture that separates the [[value function]] and [[advantage function]] into two separate streams.
- Because the [[advantage]] function can be defined as $A(s,a) = Q(s,a) - V(s)$, we can re-arrange this to get the Q-value function as $Q(s,a) = A(s,a) - V(s)$.
    - As such, we can create two networks instead of one, a $V(s)$ network and an $A(s,a)$ network, and combine them to get the Q-value function.
    - This has the effect of constraining the bias of each network, which can help with convergence.

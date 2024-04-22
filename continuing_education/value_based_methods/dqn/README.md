* What is the bellman equation? #card
  * $Q(s, a) = r + γ * max_a'(Q(s', a')) * (1 - done(s, a))$
    * $Q(s, a)$: The Q-value of taking action $a$ in state $s$
    * $r$: The reward of taking action $a$ in state $s$
    * $γ$: The discount factor
    * $max_a'(Q(s', a'))$: The maximum Q-value of the next state $s'$
    * $done(s, a)$: Whether the episode is done after taking action $a$ in state $s$
* What does the Q value represent? #card
  * The expected cumulative future reward of taking action $a$ in state $s$ following the optimal policy thereafter.
* What is the objective function of Q-learning? #card
  * $L(\theta) = \mathbb{E}[(Q_\theta(s, a) - (r + γ * max_a'(Q_\theta(s', a'))))^2]$
    * $L(\theta)$: The loss function
    * $Q(s, a)$: The Q-value of taking action $a$ in state $s$
    * $Q(s', a')$: The Q-value of taking action $a'$ in next state $s'$
    * $r$: The reward of taking action $a$ in state $s$
    * $γ$: The discount factor
* What are some differences in the `act` method between `QLearning` and `REINFORCE` #card
  * QLearning uses an argmax to select the best action whereas REINFORCE uses a softmax sample to select an action
  * QLearning has an epsilon greedy policy whereas REINFORCE has a stochastic policy
* Define epsilon greedy policy #card
  * An epsilon greedy policy is a policy that selects the best action with probability $1 - \epsilon$ and a random action with probability $\epsilon$
* What are some differences in the `train` method between `QLearning` and `REINFORCE` #card
  * QLearning trains on each step whereas REINFORCE trains at the end of each episode
  * REINFORCE needs a whole trajectory to train, because it operates on real cumulative rewards, whereas QLearning can train on each step because it operates on predicted cumulative rewards
* What are some differences in the `collect_episodes` method between `QLearning` and `REINFORCE` #card
  * QLearning uses SARS whereas REINFORCE uses SAR
  * REINFORCE needs a whole trajectory to train, because it operates on real cumulative rewards, whereas QLearning can train on each step because it operates on predicted cumulative rewards
  * This is because the bellman equation requires a mixture of one real reward and one predicted reward from the network to properly train
* What are the differences between exploration rate and temperature in Q-learning and REINFORCE? #card
  * Exploration rate is a probability of taking a random action
  * Temperature is a parameter in the softmax function that controls the stochasticity of the policy
  * Exploration either happens or does not happen and is not controlled by the neural network at all
  * Temperature just controls the output of the network. If the network is very confident in one action, it will still take that action with high probability, but if the network is unsure, it will take a random action with some probability. Therefore you don't need to decay temperature, but you do need to decay exploration rate.
* What is action replay memory? Why is it needed in QLearning but not REINFORCE. #card
    * A buffer that stores experiences for training the network.
    * Q Learning is an off-policy method, so it can learn from past experiences. REINFORCE is an on-policy method, so it can't learn from past experiences. The replay buffer is a feature of off-policy methods, not a hinderance.
* What is on-policy learning? #card
    * On-policy learning is when the agent learns from the same policy that it uses to interact with the environment.
* What is off-policy learning? #card
    * Off-policy learning is when the agent can learn from a different policy than the one it uses to interact with the environment. This allows for more efficient learning because the agent can learn from past experiences.
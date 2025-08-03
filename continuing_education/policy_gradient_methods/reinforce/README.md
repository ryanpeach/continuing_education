# Reinforce

## What is the [[REINFORCE]] scoring function?

- $L(\theta) = \frac{1}{T} \sum_{t=0}^{T-1} G_t \log \pi_{\theta}(a_t | s_t)$
    - $L(\theta)$: The loss function
    - $T$: The number of time steps in the episode
    - $G_t$: Cumulative discounted future reward at time step $t$
    - $\pi_{\theta}(a_t | s_t)$: The probability of taking action $a_t$ in state $s_t$ under [[policy]] $\pi_{\theta}$
- $\Delta \theta = \alpha r \frac{\partial \log \pi_{\theta}(s, a)}{\partial \theta}$
    - $\Delta \theta$: The update to the [[policy]] parameters. The [[laplacian]]
    - $\alpha$: The learning rate
    - $r$: The reward
    - $\partial{\log \pi_{\theta}(s, a)}$: The [[gradient]] of the [[log probability]] of taking action $a$ in state $s$ under [[policy]] $\pi_{\theta}$

## What is the [[policy gradient]] theorem?

- For any differentiable [[policy]] and for any [[policy]] objective function, the [[policy gradient]] is: $\nabla_{\theta} J(\theta) = \mathbb{E}_{\pi_{\theta}}[\nabla_{\theta} \log \pi_{\theta}(a_t | s_t) R(\tau)]$
    - $J(\theta)$: The objective function to maximize
    - $\pi_{\theta}(a_t | s_t)$: The probability of taking action $a_t$ in state $s_t$ under [[policy]] $\pi_{\theta}$
    - $R(\tau)$: The return of a trajectory $\tau$, which is often formulated as cumulative discounted future rewards.
    - $J(\theta)$: The objective function

## How does [[REINFORCE]] implement policy gradient?

Using a monte carlo method over episode rollouts.

## What is the training loop for [[REINFORCE]]?

- Initialize the [[policy]] $\pi_{\theta}$ with random weights
- For each episode:
    - Generate a trajectory $\tau$ by following the [[policy]] $\pi_{\theta}$
    - Compute the return $R(\tau)$
    - Compute the [[policy gradient]] $\nabla_{\theta} J(\theta)$
    - Update the [[policy]] parameters $\theta$ with the [[gradient]]

## What is $\pi_{\theta}(a | s)$?

- The function which is learned by the [[REINFORCE]] algorithm
- The probability of taking action $a$ in state $s$ under [[policy]] $\pi_{\theta}$

## Why do you need to normalize the rewards in [[REINFORCE]]?

Since the rewards are arbitrary and directly part of the objective/loss function, they are normalized to make the optimization more stable.

## What is a multinomial distribution?

A probability distribution over a discrete number of possible outcomes, where each outcome has a probability associated with it. Like a dice roll, where each face has a probability of being rolled.

## What are some advantages of [[policy gradient]] methods over [[value]] based methods?

- They can learn stochastic policies
- They can learn policies in high-dimensional or continuous action spaces
- They can have better convergence properties, they can be made to change smoothly over time with sampling rather than depending on an argmax operation.

## What are some disadvantages of [[policy gradient]] methods vs [[value]] based methods ?

- They have high variance in the [[gradient]]s which can lead to
    - Slow convergence
    - Catastrophic forgetting
- They can be sensitive to the choice of step size
- They can be computationally expensive

## What is the softmax equation with temperature?

- $P(i) = \frac{e^{Z_i/T}}{\sum_{j} e^{Z_j/T}}$
- $P(i)$: The probability of outcome $i$
- $Z_i$: The logit of outcome $i$
- $T$: The temperature parameter

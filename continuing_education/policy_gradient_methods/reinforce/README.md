# Logseq flash cards

* What is the REINFORCE scoring function? #card
  * $L(\theta) = \frac{1}{T} \sum_{t=0}^{T-1} G_t \log \pi_{\theta}(a_t | s_t)$
    * $L(\theta)$: The loss function
    * $T$: The number of time steps in the episode
    * $G_t$: Cumulative discounted future reward at time step $t$
    * $\pi_{\theta}(a_t | s_t)$: The probability of taking action $a_t$ in state $s_t$ under policy $\pi_{\theta}$
  * $\Delta \theta = \alpha r \frac{\partial \log \pi_{\theta}(s, a)}{\partial \theta}$
    * $\Delta \theta$: The update to the policy parameters
      * $\Delta$ is the first or second gradient? What is it called? #card
        * Second
        * Laplacian operator
        * What is the difference between a Henessian and a Laplacian? #card
            * Henessians produce a matrix of second derivatives, while Laplacians produce a scalar. Both are second order derivatives.
    * $\alpha$: The learning rate
    * $r$: The reward
    * $\partial{\log \pi_{\theta}(s, a)}$: The gradient of the log probability of taking action $a$ in state $s$ under policy $\pi_{\theta}$
* What is the policy gradient theorem? #card
    * For any differentiable policy and for any policy objective function, the policy gradient is: $\nabla_{\theta} J(\theta) = \mathbb{E}_{\pi_{\theta}}[\nabla_{\theta} \log \pi_{\theta}(a_t | s_t) R(\tau)]$
        * What does the symbol $\nabla$ mean? #card
            * The gradient, which is a vector of single partial derivatives in each dimension of the input space.
        * $J(\theta)$: The objective function to maximize
        * $\pi_{\theta}(a_t | s_t)$: The probability of taking action $a_t$ in state $s_t$ under policy $\pi_{\theta}$
        * $R(\tau)$: The return of a trajectory $\tau$, which is often formulated as cumulative discounted future rewards.
        * How is this different from the REINFORCE scoring function? #card
            * It's more general
            * It's formulated as an expectation over a gradient
            * It's a maximization rather than a loss
            * But very similar to the REINFORCE scoring function
    * $J(\theta)$: The objective function
* What is $\pi_{\theta}(a | s)$? #card
    * The probability of taking action $a$ in state $s$ under policy $\pi_{\theta}$
* What is a multinomial distribution? #card
    * A probability distribution over a discrete number of possible outcomes, where each outcome has a probability associated with it. Like a dice roll, where each face has a probability of being rolled.
* What are some advantages of policy gradient methods over value-based methods? #card
    * They can learn stochastic policies
    * They can learn policies in high-dimensional or continuous action spaces
    * They can have better convergence properties, they can be made to change smoothly over time with sampling rather than depending on an argmax operation.
* What are some disadvantages of policy gradient methods vs value-based methods? #card
    * They have high variance in the gradients which can lead to # card
        * Slow convergence
        * Catastrophic forgetting
    * They can be sensitive to the choice of step size
    * They can be computationally expensive
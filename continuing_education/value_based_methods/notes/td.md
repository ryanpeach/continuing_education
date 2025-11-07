# Temporal Difference

## What is Temporal Difference (TD) learning?

Temporal difference ([[TD]]) learning refers to a class of model-free reinforcement learning methods which learn by bootstrapping from the current estimate of the value function. These methods sample from the environment, like [[Monte Carlo]] methods, and perform updates based on current estimates, like dynamic programming methods.

https://en.wikipedia.org/wiki/Actor-critic_algorithm

## What is the difference between TD learning and Monte Carlo methods?

While [[Monte Carlo]] methods only adjust their estimates once the final outcome is known, [[TD]] methods adjust predictions to match later, more accurate, predictions about the future before the final outcome is known. This is a form of [[bootstrapping]].

source: https://en.wikipedia.org/wiki/Temporal_difference_learning

## What are the key features of TD learning?

- **[[Bootstrapping]]**: [[TD]] methods update estimates based on other learned estimates, rather than waiting for the final outcome.
- **Model-free**: [[TD]] learning does not require a model of the environment, making it
suitable for environments where the dynamics are unknown.
- **Online learning**: [[TD]] methods can learn from each step of interaction with the environment, allowing for continuous updates and learning.
- **Temporal credit assignment**: [[TD]] learning assigns credit to actions based on their contribution to future rewards, allowing for more efficient learning in environments with delayed rewards.

source: AI

## What is the SARSA algorithm?


## What is the TD(0) algorithm?


## What is the $TD\lambda$ algorithm?


## Biological Inspiration of TD Learning

The [[TD]] algorithm has also received attention in the field of neuroscience. Researchers discovered that the firing rate of dopamine neurons in the ventral tegmental area (VTA) and substantia nigra (SNc) appear to mimic the error function in the algorithm. The error function reports back the difference between the estimated reward at any given state or time step and the actual reward received. The larger the error function, the larger the difference between the expected and actual reward. When this is paired with a stimulus that accurately reflects a future reward, the error can be used to associate the stimulus with the future reward.

Dopamine cells appear to behave in a similar manner. In one experiment measurements of dopamine cells were made while training a monkey to associate a stimulus with the reward of juice. Initially the dopamine cells increased firing rates when the monkey received juice, indicating a difference in expected and actual rewards. Over time this increase in firing back propagated to the earliest reliable stimulus for the reward. Once the monkey was fully trained, there was no increase in firing rate upon presentation of the predicted reward. Subsequently, the firing rate for the dopamine cells decreased below normal activation when the expected reward was not produced. This mimics closely how the error function in [[TD]] is used for reinforcement learning.

The relationship between the model and potential neurological function has produced research attempting to use [[TD]] to explain many aspects of behavioral research. It has also been used to study conditions such as schizophrenia or the consequences of pharmacological manipulations of dopamine on learning.

https://en.wikipedia.org/wiki/Temporal_difference_learning#In_neuroscience

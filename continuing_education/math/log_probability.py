# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: continuing-education-vJKa4-To-py3.10
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # Log Probabilities and Likelihoods
#
# Log probabilities and likelihoods come up a lot in machine learning, and I'm constantly
# referring back to their definitions. So I thought I'd write a quick note on them.
#
# First lets just plot a log function from 0 to 1.

# %%
import plotly.express as px
import numpy as np


def plot_logarithm() -> None:
    # Generating values from 0.01 to 1 (avoiding zero to prevent -inf in log calculation)
    x_values = np.linspace(0.001, 1, 400)
    log_values = np.log(x_values)

    # Creating the plot
    fig = px.line(
        x=x_values,
        y=log_values,
        labels={"x": "x", "y": "log(x)"},
        title="Logarithm of x from 0.01 to 1",
    )
    fig.update_xaxes(range=[0, 1])
    fig.show()


if __name__ == "__main__":
    plot_logarithm()

# %% [markdown]
# It's asymptotic to negative infinity at 0, and equal to 0 at 1.
#
# The key insight is that since probabilities infinitely approach but rarely equal 0 or 1, the log of a probability amplifies both extremes. Probabilities very close to 1 will similarly be very close to 0 in log space, and probabilities very close to 0 will be very negative in log space.
#
# ## Equalities and Computational Efficiency
#
# Next lets remember some mathematical properties of logs.
# * They turn multiplication into addition: $\log(a \cdot b) = \log(a) + \log(b)$
# * They turn division into subtraction: $\log(a / b) = \log(a) - \log(b)$
# * They turn exponentiation into multiplication: $\log(a^b) = b \cdot \log(a)$

# %%
from sympy import symbols, log, Eq
from sympy.simplify import simplify


def logarithm_properties() -> None:
    # Define the symbols
    a, b = symbols("a b", positive=True, real=True)

    # Logarithmic properties
    log_mult = Eq(log(a * b), log(a) + log(b))
    log_div = Eq(log(a / b), log(a) - log(b))
    log_exp = Eq(log(a**b), b * log(a))

    # Check and simplify each equation to verify correctness
    simplify_log_mult = simplify(log_mult.lhs - log_mult.rhs)
    simplify_log_div = simplify(log_div.lhs - log_div.rhs)
    simplify_log_exp = simplify(log_exp.lhs - log_exp.rhs)

    assert simplify_log_mult == 0
    assert simplify_log_div == 0
    assert simplify_log_exp == 0
    print("All proposed logarithmic properties are correct!")


if __name__ == "__main__":
    logarithm_properties()


# %% [markdown]
# These properties are useful in machine learning because they allow us to turn products of probabilities into sums of log probabilities, which are easier computationally.
#

# %% [markdown]
# ## Loss Functions
#
# Finally, when we use a log probability in a loss function, because its asymptotic at P(0) and 0 at P(1) it has a much better gradient than a regular probability. This is why we often use log probabilities in loss functions.
#
# For example look at the REINFORCE score function:
#
# $J(\theta) = \frac{1}{T} \sum_{t=0}^{T-1} G_t \log \pi_{\theta}(a_t | s_t)$
#
# Simplified:
#
# $J(\theta) = \sum_{t=0}^{T-1} G_t \log \pi_{\theta}(a_t | s_t)$
#
# The loss function (minimized by gradient descent) becomes:
#
# $L(\theta) = \sum_{t=0}^{T-1} G_t (-\log \pi_{\theta}(a_t | s_t))$
#
# Where $G_t$ is the cumulative discounted future reward at time $t$, and $\pi_{\theta}(a_t | s_t)$ is the probability of taking action $a_t$ in state $s_t$ under policy $\pi_{\theta}$.
#
# Lets make a table relating $\pi_{\theta}(a_t | s_t)$ to the reward $G_t$ assuming the possible rewards are 0 and 1:
#
#
# | $\pi_{\theta}(a_t \| s_t)$ | $G_t$ | $-G_t \log \pi_{\theta}(a_t \| s_t)$ |
# |:-------------------------:|:-----:|:-----------------------------------:|
# | 0.01                      | 1.0   |  2                                  |
# | 0.1                       | 1.0   |  1                                  |
# | 0.5                       | 1.0   |  0.301                              |
# | 0.9                       | 1.0   |  0.045                              |
# | 0.99                      | 1.0   |  0.0043                             |
#
#
# And 0 whenever $G_t$ is 0. Assuming that's the negative reward.
#
# What this tells us is that the loss function will get asymptotically closer to 0 as the probability of the action taken approaches 1. It will also explode into the positives as the probability of the action taken approaches 0. For positive rewards, this is good because it will push the probability of actions which lead to positive rewards towards 1.
#

# %% [markdown]
# One way you can visualize this is by mapping the [0, 1] range of probabilities to the [-inf, inf] continuous numbers using a sigmoid function.
#
# $y = \log(\frac{1}{1+e^{-x}})$


# %%
def plot_logarithm_continuous() -> None:
    # Generating values from 0.01 to 1 (avoiding zero to prevent -inf in log calculation)
    max_x = 5
    x_values = np.linspace(-max_x, max_x, 1000)
    log_values = np.log(1 / (1 + np.exp(-x_values)))

    # Creating the plot
    fig = px.line(
        x=x_values,
        y=log_values,
        labels={"x": "x", "y": "log(x/(1+e^(-x)))"},
        title=f"Logarithm of x from {1/(1+np.exp(-max_x))*100:.2f}% to {1/(1+np.exp(max_x))*100:.2f}% where x is first passed through a sigmoid function to make it continuous",
    )
    fig.show()


if __name__ == "__main__":
    plot_logarithm_continuous()

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

# %% [markdown]
# # Log Probabilities and Likelihood
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

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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import sympy as sp
from sympy.physics.units import hour

# %%
sp.init_printing()

# %% [markdown]
# # Farmer + Hoe Thought Experiment
#
# Assume there is a farmer which can do one of three things each with different labor times under different conditions:
#
# 1. Produce a hoe
# 2. Farm land to produce corn
#
# A hoe can be made of sticks and stones and both are plentiful, no original Means of Production (MoP) required. Similarly land is plentiful so the farm land is not burdened by the cost of existing property ownership.
#
# The farmer must produce a certain amount of corn per timeperiod or he will die. He need not produce extra. He prefers to spend as little time working as possible.
#
# ## Symbols
#
# * Lets call the labor time required to produce a machine (like the hoe) $m$ without using any means of production (represented by $_0$) $L_0(m)$.
#   * In the analogy, this would be the labor required to produce a hoe $L_0(hoe)$
# * Lets also call the labor time required to produce a final product (like corn) $x$ without using any MoP $L_0(x)$.
#   * So for corn this would be $L_0(corn)$
# * Lets call the labor time required to produce a final product (like corn) given a machine (like a hoe) $L_m(x)$
# * Lets call the lifetime of a machine $m$ in total units of production of the final product $x$ $T_x(m)$

# %%
corn = sp.Symbol("corn")
hoe = sp.Symbol("hoe")
L_0 = sp.Function("L_0")
L_0_of_hoe = L_0(hoe) * hour / hoe
L_0_of_corn = L_0(corn) * hour / corn
L_hoe = sp.Function("L_{hoe}")
L_hoe_of_corn = L_hoe(corn) * hour / corn
T_corn = sp.Function("T_{corn}")
T_corn_of_hoe = T_corn(hoe) * corn / hoe
L_0_of_hoe, L_0_of_corn, L_hoe_of_corn, T_corn_of_hoe

# %% [markdown]
# ## Assumptions
#
# 1. ~~The farmer needs 1 corn per $D$ where D<L_0(corn)~~ Was unneded
# 2. One x is produced per $L(x)$ (no sub necessary, it applies to both)
# 3. $L_m(x) < L_0(x)$ (MoP always saves time)

# %% [markdown]
# # First Experiment
#
# First lets find the quantity of total labor time of the farmer first producing a hoe, then using it to produce corn, until the lifetime of the hoe is used up.
#
# The time it takes the farmer to make the hoe is $L_0(hoe)$, the time it takes to make the corn with the hoe is $L_{hoe}(corn)$. Thus the total labor time over one hoe lifetime is $L_0(hoe)+L_{hoe}(corn)*T_{corn}(hoe)$.

# %%
first_experiment_labor_time = sp.simplify((L_0_of_hoe + L_hoe_of_corn * T_corn_of_hoe))
first_experiment_labor_time

# %% [markdown]
# # Second Experiment
#
# Second lets find the quantity of total labor time of the farmer in the same amount of living time as the first experiment.

# %%
second_experiment_labor_time = sp.simplify(L_0_of_corn * T_corn_of_hoe)
second_experiment_labor_time

# %% [markdown]
# # Questions
#
# 1. How much labor did the hoe save the farmer?

# %%
answer_q1 = sp.simplify(
    (second_experiment_labor_time - first_experiment_labor_time) * hoe
)
answer_q1

# %% [markdown]
# 2. How much labor did the hoe cost?

# %%
answer_q2 = L_0_of_hoe * hoe
answer_q2

# %% [markdown]
# 3. Under what conditions did the hoe save more labor than it cost?

# %%
answer_q3 = sp.simplify(answer_q1 / hour > answer_q2 / hour)
answer_q3

# %% [markdown]
# 4. Is this true given the assumptions?

# %%
# Subtract both sides by $L_0(hoe)$ and simplify
c = L_0_of_hoe * hoe / hour
step1 = answer_q3.lhs + c < answer_q3.rhs + c
step1

# %%
# Since T_corn(hoe) is a positive factor, it does not effect the inequality, we can cancel it out
step2 = step1.subs(T_corn_of_hoe * hoe / corn, 1)
step2

# %% [markdown]
# There you have it, the hoe saves more labor than the labor it took to create it iff the difference in corn production time is at least 2 times the hoe production time. The farmer would not produce the hoe otherwise, given "He prefers to spend as little time working as possible."

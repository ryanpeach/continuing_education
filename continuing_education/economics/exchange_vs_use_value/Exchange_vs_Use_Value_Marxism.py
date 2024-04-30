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
from sympy.physics.units import hour, Unit

# %%
sp.init_printing()

# %% [markdown]
# # Symbols
#
# * Assume two capitalist running two companies A, B.
# * Company A sells machines. Machines have:
#   * A factor $n$ which is the number of units of production B per machine (unit of production A).
#   * $n$ == 0 if the machine is not working.
#   * Company B buys from company A one machine and uses it to produce units of production B.
# * The **labor time (labor value)** in company x is $X_x$ measured in hours per unit of production. It can vary given number of machines $N_x$.
# * The **wage (labor power)** for the employees at company x is $Y_x$ measured in dollars per hour. It can vary given number of machines $N_x$.
# * The **non wage expenses (constant capital)** of company x is $I_x$ measured in dollars per unit of production. It can vary given number of machines $N_x$.
# * The **selling price (exchange value)** for the final product in company x is $Z_x$ measured in dollars per unit of production. It can vary given number of machines $N_x$.
# * The **profit** for company x is $K_x$ measured in dollars per unit of production. It can vary given number of machines $N_x$.
# * Assume A produces a machine which when bought by B has a **labor saving factor** of $\theta(N_x)$, which is multiplied by the labor time of B to produce the new labor time of per unit of production of B.

# %%
dollar = Unit("\$")
dollar

# %%
from collections import namedtuple

lam = sp.Symbol("\lambda") * dollar / hour

Company = namedtuple(
    "Company",
    [
        "labor_time",
        "wage",
        "selling_price",
        "non_wage_expenses",
        "profit",
        "unit",
    ],
)
unit_A = sp.Symbol("unit_A")
A = Company(
    labor_time=sp.Symbol("X_A") * hour / unit_A,
    wage=sp.Symbol("Y_A") * dollar / hour,
    selling_price=sp.Symbol("Z_A") * dollar / unit_A,
    non_wage_expenses=sp.Symbol("I_A") * dollar / unit_A,
    profit=sp.Symbol("K_A") * dollar / unit_A,
    unit=unit_A,
)
del unit_A  # Use the definition in the tuple
unit_B = sp.Symbol("unit_B")
n_unitless = sp.Symbol("n")
B = Company(
    labor_time=sp.Function("X_B")(n_unitless) * hour / unit_B,
    wage=sp.Function("Y_B")(n_unitless) * dollar / hour,
    selling_price=sp.Function("Z_B")(n_unitless) * dollar / unit_B,
    non_wage_expenses=sp.Function("I_B")(n_unitless) * dollar / unit_B,
    profit=sp.Function("K_B")(n_unitless) * dollar / unit_B,
    unit=unit_B,
)
n = n_unitless * B.unit / A.unit  # Now give n units
del unit_B  # Use the definition in the tuple
tuple(A), tuple(B), n

# %% [markdown]
# # Definitions
#
# 1. The cost of production per unit for company x == labor cost per unit + other cost per unit


# %%
def total_cost_of_production(x: str):
    comp = A if x == "A" else B
    return comp.wage * comp.labor_time + comp.non_wage_expenses


total_cost_of_production("A"), total_cost_of_production("B")


# %% [markdown]
# 2. The profit to the capitalist is the difference in the selling price and the total cost of expenses


# %%
def profit_perspective_of_point_of_sale(x: str):
    comp = A if x == "A" else B
    return sp.Eq(comp.profit, comp.selling_price - total_cost_of_production(x))


profit_perspective_of_point_of_sale("A"), profit_perspective_of_point_of_sale("B")

# %% [markdown]
# 3. The difference between the non wage cost of production for company B before the machine and the non wage cost of production after the machine is equal to the cost of the machine (sold by company A)

# %%
delta_non_wage_expenses_B = sp.Eq(
    sp.diff(B.non_wage_expenses, n_unitless), A.selling_price
)
delta_non_wage_expenses_B

# %% [markdown]
# # Assumptions

# %% [markdown]
# 1. Under max's theory of surplus value, profit is labor value (labor time converted into exchange value) minus labor power (wage)


# %%
def profit_perspective_of_surplus_value(x: str):
    comp = A if x == "A" else B
    return sp.Eq(
        comp.profit, sp.simplify(comp.labor_time * lam - comp.wage * comp.labor_time)
    )


profit_perspective_of_surplus_value("A"), profit_perspective_of_surplus_value("B")

# %% [markdown]
# 2.

# %% [markdown]
# # Methodology

# %% [markdown]
# Because we are really talking about the change in profit, change in labor, etc, this will only be solved with differentials.


# %%
def delta_total_cost_of_production(x: str):
    comp = A if x == "A" else B
    return sp.diff(comp.non_wage_expenses, n_unitless)


delta_total_cost_of_production("x")


# %%
def delta_profit_perspective_of_point_of_sale(x: str):
    eq = profit_perspective_of_point_of_sale(x)
    return sp.Eq(sp.diff(eq.lhs, n_unitless), sp.diff(eq.rhs, n_unitless))


delta_profit_perspective_of_point_of_sale("B")

# %% [markdown]
# Substitute out that $\Delta{I_B}$ using Equality #3

# %%
delta_profit_perspective_of_point_of_sale_B_sub = sp.Eq(
    delta_profit_perspective_of_point_of_sale("B").lhs,
    delta_profit_perspective_of_point_of_sale("B").rhs.subs(
        delta_non_wage_expenses_B.lhs / dollar * B.unit,
        delta_non_wage_expenses_B.rhs / dollar * B.unit,
    ),
)
delta_profit_perspective_of_point_of_sale_B_sub


# %%
def delta_profit_perspective_of_surplus_value(x: str):
    eq = profit_perspective_of_surplus_value(x)
    return sp.Eq(sp.diff(eq.lhs, n_unitless), sp.diff(eq.rhs, n_unitless))


delta_profit_perspective_of_surplus_value_B = delta_profit_perspective_of_surplus_value(
    "B"
)
delta_profit_perspective_of_surplus_value_B

# %% [markdown]
# Take the difference of the right hand side of both.
#
# Because we did these differentials differently, taking their difference will relate them by what varies between them.
#
# In this case we are specifically interested in relating the change in non-wage expenses to the change in labor time, from the perspective of the marxist theory of value.

# %%
zero_eq = sp.Eq(
    delta_profit_perspective_of_surplus_value_B.lhs
    - delta_profit_perspective_of_point_of_sale_B_sub.lhs,
    delta_profit_perspective_of_surplus_value_B.rhs
    - delta_profit_perspective_of_point_of_sale_B_sub.rhs,
)
zero_eq

# %% [markdown]
# One way to interpret this is that $\lambda$ (which is a highly speculative variable assuming all labor has a fixed exchange value per unit time) is equal to the difference between the wage and the ratio of the price of the machine to the labor saving power of the machine.

# %%
sp.Eq(lam / (dollar / hour), sp.expand(sp.solve(zero_eq.rhs, lam / (dollar / hour))[0]))

# %% [markdown]
# Another way of looking at this is to say that the marxist definition of profit being extraction of surplus value from the worker is only true when this equality is true:

# %%
delta_x_b = sp.Eq(
    sp.diff(B.labor_time, n_unitless),
    sp.expand(sp.solve(zero_eq, sp.diff(B.labor_time, n_unitless) * B.unit / hour)[0]),
)
delta_x_b

# %% [markdown]
# Move lambda to the left side

# %%
sp.Eq(
    delta_x_b.lhs * lam / dollar * hour,
    sp.expand(sp.simplify(delta_x_b.rhs * lam / dollar * hour)),
)

# %% [markdown]
#
# So:
#
#
#
# Other Research Questions:
# * $\lambda$ could be loosened a bit to have a different conversion factor for each company. Different areas of work could have different "value". See what falls out of that model vs the current one.
# * You can get rid of lambda and just use wage if you do not believe in LtV. The point of sale equation can be differentiated along the change in labor time dimension to eliminate the $-\lambda$ term in the final result.
# * What is the difference between `use-value - labor-value` as a factor of exploitation vs the more traditional marxist exploitation quantity of labor-value - labor-power? What if I used the product to do as the other capitalist does instead of selling it to them? Would that fit under roemer's exploitation theory?

# %% [markdown]
# That seems perfectly reasonable. So given the assumptions are true, the surplus value theory of profix is compatible with the capitalist definition of profit.

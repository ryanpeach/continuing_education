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
from sympy.physics.units import hour, Dimension, Unit, Quantity

# %%
sp.init_printing()

# %% [markdown]
# # Symbols
#
# * Assume two capitalist running two companies A, B.
# * The **labor time (labor value)** in company x is $X_x$ measured in hours per unit of production.
# * The **wage (labor power)** for the employees at company x is $Y_x$ measured in dollars per hour.
# * The **non wage expenses (constant capital)** of company x is $I_x$ measured in dollars per unit of production.
# * The **selling price (exchange value)** for the final product in company x is $Z_x$ measured in dollars per unit of production.
# * The **profit** for company x is $K_x$ measured in dollars per unit of production
# * Assume A produces a machine which when bought by B has a **labor saving power (use value)** of $\hat{X}_B$ hours per unit of production.

# %%
dollar = Unit("\$")
unit = Unit("unit")
dollar, unit

# %%
from collections import namedtuple
from functools import cache
Company = namedtuple('Company', ['labor_time', 'wage', 'selling_price', 'non_wage_expenses', 'labor_saving_value', 'profit'])

@cache
def company(x: str) -> Company:
    Xx, Yx, Zx, Ix, Kx, Xhatx = sp.symbols(f'X_{x}, Y_{x}, Z_{x}, I_{x}, K_{x}, \hat{{X}}_{x}')
    labor_time = Xx*hour/unit
    wage = Yx*dollar/hour
    selling_price = Zx*dollar/unit
    non_wage_expenses = Ix*dollar/unit
    labor_saving_value = Xhatx*hour/unit
    profit = Kx*dollar/unit
    return Company(labor_time, wage, selling_price, non_wage_expenses, labor_saving_value, profit)


# %%
tuple(company("x"))


# %% [markdown]
# # Equalities
#
# 1. The cost of production per unit for company x == labor cost per unit + other cost per unit

# %%
def total_cost_of_production(x: str):
    comp = company(x)
    return comp.wage*comp.labor_time+comp.non_wage_expenses
total_cost_of_production('x')


# %% [markdown]
# 2. The profit to the capitalist is the difference in the selling price and the total cost of expenses

# %%
def profit_perspective_of_point_of_sale(x: str):
    comp = company(x)
    return sp.Eq(comp.profit, comp.selling_price-total_cost_of_production(x))
profit_perspective_of_point_of_sale('x')

# %% [markdown]
# # Assumptions
#
# 1. Under marx's LtV, the exchange value of a good equals its labor value times some conversion factor $\lambda$ measured in dollars per hour.

# %%
lam = sp.Symbol('\lambda')*dollar/hour

def LtV(x: str):
    comp = company(x)
    return sp.Eq(comp.selling_price, comp.labor_time*lam)
LtV('x')


# %% [markdown]
# 2. Under max's theory of surplus value, profit is labor value (labor time converted into exchange value) minus labor power (wage)

# %%
def profit_perspective_of_surplus_value(x: str):
    comp = company(x)
    return sp.Eq(comp.profit, sp.simplify(comp.labor_time*lam-comp.wage*comp.labor_time))
profit_perspective_of_surplus_value('x')

# %% [markdown]
# 3. Production capacity remains constant.
#
# Under this assumption, we can derive that the labor saved by the machine will equal the labor cut (fired) by the capitalist.

# %%
B = company('B') # Since only company B uses the machine

# %%
labor_cut_by_B = B.labor_saving_value
labor_cut_by_B

# %%
new_labor_time_B = B.labor_time - labor_cut_by_B
new_labor_time_B

# %%
company('B').labor_time/hour*unit

# %%
new_profit_perspective_of_surplus_value_B = profit_perspective_of_surplus_value('B').rhs.subs(company('B').labor_time/hour*unit, new_labor_time_B/hour*unit)
new_profit_perspective_of_surplus_value_B = sp.Eq(profit_perspective_of_surplus_value('B').lhs, sp.simplify(new_profit_perspective_of_surplus_value_B))
new_profit_perspective_of_surplus_value_B

# %%
new_profit_perspective_of_point_of_sale_B = profit_perspective_of_point_of_sale('B').rhs.subs(company('B').labor_time/hour*unit, new_labor_time_B/hour*unit)
new_profit_perspective_of_point_of_sale_B = sp.Eq(profit_perspective_of_point_of_sale('B').lhs, sp.simplify(new_profit_perspective_of_point_of_sale_B))
new_profit_perspective_of_point_of_sale_B

# %% [markdown]
# # Conclusions
#
# The difference in the perspective of profit as point of sale vs surplus value under conditions of LtV
#
# I wonder what this could mean?

# %%
sp.simplify(new_profit_perspective_of_point_of_sale_B.rhs - new_profit_perspective_of_surplus_value_B.rhs)

# %%

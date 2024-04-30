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
# * The **labor time (labor value)** in company x is $X_x$ measured in hours per unit of production.
# * The **wage (labor power)** for the employees at company x is $Y_x$ measured in dollars per hour.
# * The **non wage expenses (constant capital)** of company x is $I_x$ measured in dollars per unit of production.
# * The **selling price (exchange value)** for the final product in company x is $Z_x$ measured in dollars per unit of production.
# * The **profit** for company x is $K_x$ measured in dollars per unit of production
# * Assume A produces a machine which when bought by B has a **labor saving power (use value)** of $\Delta{X_B}$ hours per unit of production.

# %%
dollar = Unit("\$")
dollar

# %%
from collections import namedtuple
from functools import cache


@cache
def delta(x: sp.Symbol, units: sp.Expr):
    symbol_itself = x / units
    new_symbol = sp.Symbol(f"\Delta{{{symbol_itself}}}")
    return new_symbol * units


Company = namedtuple(
    "Company",
    [
        "labor_time",
        "wage",
        "selling_price",
        "non_wage_expenses",
        "labor_saving_value",
        "profit",
        "unit",
    ],
)


@cache
def company(x: str) -> Company:
    Xx, Yx, Zx, Ix, Kx = sp.symbols(f"X_{x}, Y_{x}, Z_{x}, I_{x}, K_{x}")
    unit = sp.Symbol(f"unit_{x}")
    labor_time = Xx * hour / unit
    wage = Yx * dollar / hour
    selling_price = Zx * dollar / unit
    non_wage_expenses = Ix * dollar / unit
    labor_saving_value = delta(Xx * hour / unit, units=hour / unit)
    profit = Kx * dollar / unit
    return Company(
        labor_time,
        wage,
        selling_price,
        non_wage_expenses,
        labor_saving_value,
        profit,
        unit,
    )


# %%
tuple(company("x"))


# %% [markdown]
# # Definitions
#
# 1. The cost of production per unit for company x == labor cost per unit + other cost per unit


# %%
def total_cost_of_production(x: str):
    comp = company(x)
    return comp.wage * comp.labor_time + comp.non_wage_expenses


total_cost_of_production("x")


# %% [markdown]
# 2. The profit to the capitalist is the difference in the selling price and the total cost of expenses


# %%
def profit_perspective_of_point_of_sale(x: str):
    comp = company(x)
    return sp.Eq(comp.profit, comp.selling_price - total_cost_of_production(x))


profit_perspective_of_point_of_sale("x")

# %% [markdown]
# 3. The difference between the non wage cost of production for company B before the machine and the non wage cost of production after the machine is equal to the cost of the machine (sold by company A)

# %%
delta_non_wage_expenses_B = sp.Eq(
    delta(company("B").non_wage_expenses, units=dollar / company("B").unit),
    company("A").selling_price,
)
delta_non_wage_expenses_B

# %% [markdown]
# # Assumptions
#
# 1. Under marx's LtV, the exchange value of a good equals its labor value times some conversion factor $\lambda$ measured in dollars per hour.

# %%
lam = sp.Symbol("\lambda") * dollar / hour


def LtV(x: str):
    comp = company(x)
    return sp.Eq(comp.selling_price, comp.labor_time * lam)


LtV("x")


# %% [markdown]
# 2. Under max's theory of surplus value, profit is labor value (labor time converted into exchange value) minus labor power (wage)


# %%
def profit_perspective_of_surplus_value(x: str):
    comp = company(x)
    return sp.Eq(
        comp.profit, sp.simplify(comp.labor_time * lam - comp.wage * comp.labor_time)
    )


profit_perspective_of_surplus_value("x")

# %% [markdown]
# 3. Production capacity remains constant.
#
# Under this assumption, we can derive that the labor saved by the machine will equal the labor cut (fired) by the capitalist.

# %%
B = company("B")  # Since only company B uses the machine

# %%
labor_cut_by_B = B.labor_saving_value
labor_cut_by_B

# %%
new_labor_time_B = B.labor_time - labor_cut_by_B
new_labor_time_B


# %% [markdown]
# # Methodology

# %% [markdown]
# Because we are really talking about the change in profit, change in labor, etc, this will only be solved with differentials.


# %%
def delta_total_cost_of_production(x: str):
    comp = company(x)
    return comp.wage * delta(comp.labor_time, units=hour / comp.unit) + delta(
        comp.non_wage_expenses, units=dollar / comp.unit
    )


delta_total_cost_of_production("x")


# %%
def delta_profit_perspective_of_point_of_sale(x: str):
    comp = company(x)
    return sp.Eq(
        delta(comp.profit, units=dollar / comp.unit),
        delta(comp.selling_price, units=dollar / comp.unit)
        - delta_total_cost_of_production(x),
    )


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
    comp = company(x)
    return sp.Eq(
        delta(comp.profit, units=dollar / comp.unit),
        sp.expand(
            delta(comp.labor_time, units=hour / comp.unit) * lam
            - comp.wage * delta(comp.labor_time, units=hour / comp.unit)
        ),
    )


delta_profit_perspective_of_surplus_value_B = delta_profit_perspective_of_surplus_value(
    "B"
)
delta_profit_perspective_of_surplus_value_B

# %% [markdown]
# Now take the difference of these two interpretations of delta profit.
#
# This says the price of the machine plus the the change in labor converted to dollars per unit minus the change in the sale price of the final good is zero.

# %%
zero_eq = sp.Eq(
    delta_profit_perspective_of_surplus_value_B.lhs
    - delta_profit_perspective_of_point_of_sale_B_sub.lhs,
    delta_profit_perspective_of_surplus_value_B.rhs
    - delta_profit_perspective_of_point_of_sale_B_sub.rhs,
)
zero_eq

# %% [markdown]
# So why aren't these equal? It's basically saying that these are equivalent when $-Z_A+\Delta{Z_B}=\Delta{X_B}*\lambda$

# %% [markdown]
# One way to interpret this is that $\lambda$ (which is a highly speculative variable assuming all labor has a fixed exchange value per unit time) is equal somehow to the difference in the ratio of the price of the machine and the change in labor of the workers in units of final product per machine, and the ratio of the change in the price of the final product over the change in hours of the worker. Take that for what you will, it's a bit complicated for my taste.

# %%
sp.Eq(lam / (dollar / hour), sp.expand(sp.solve(zero_eq.rhs, lam / (dollar / hour))[0]))

# %% [markdown]
# Another way of looking at this is to say that the marxist definition of profit being extraction of surplus value from the worker is only true when this equality is true:

# %%
delta_x_b = sp.Eq(
    company("B").labor_saving_value / (hour / B.unit),
    sp.expand(sp.solve(zero_eq, company("B").labor_saving_value / (hour / B.unit))[0]),
)
delta_x_b

# %% [markdown]
# Moving lambda to the left for ease of reading

# %%
sp.Eq(
    delta_x_b.lhs * lam * (hour / dollar),
    sp.simplify(delta_x_b.rhs * lam * (hour / dollar)),
)

# %% [markdown]
# This says the marxist definition of profit will be true iff the difference in the labor hours of the employees times the conversion factor into dollars is equal to the difference between the cost of the machine scaled by the ratio of final product per machine (probably ametorized) and the change in price of the final product.
#
# So:
#
# * If you increase prices to offset the cost of the machine then you don't need to reduce working hours.
# * If you reduce working hours to offset the cost of the machine then you don't need to increase prices.

# %% [markdown]
# That seems perfectly reasonable. So given the assumptions are true, the surplus value theory of profix is compatible with the capitalist definition of profit.

# %%

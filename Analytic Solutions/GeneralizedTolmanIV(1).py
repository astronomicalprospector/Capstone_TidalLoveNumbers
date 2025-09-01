import os
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

output_filename = "AnalyticSolution_GeneralizedTolmanIV(1).csv"

df = pd.DataFrame(columns=['compactness', 'k2'])
existing_indices = []

#Analytic Functions (Generalized Tolman IV Solution)
def scale_pressure(beta, x):
    p_num = (1 - 3 * beta) * (1 - x)
    p_den = (2 - 3 * beta) * (1 - 3 * beta + 2 * beta * x)
    frac = beta * p_num / p_den
    return frac

def scale_density(beta, x):
    rho_num = (1 - 3 * beta) * ((2 - 3 * beta) * (1 - 3 * beta) + beta * (3 - 7 * beta) * x + (2 * beta**2 * x**2))
    rho_den = (2 - 3 * beta) * (1 - 3 * beta + 2 * beta * x)**2
    frac = rho_num / rho_den
    return frac

def squared_sound_speed(beta, x):
    c_num = 1 - 3 * beta + 2 * beta * x
    c_den = 5 - 15 * beta + 2 * beta * x
    return c_num / c_den

#metric functions (used in differentials)
def exp_lambda(beta, x): # dimensionless
    e_num = 1 - 3 * beta + 2 * beta * x
    e_den = (1 - 3 * beta + beta * x) * (1 - beta * x)
    return e_num / e_den

def v_prime(alpha, beta, x): # dimensionless
    return exp_lambda(beta, x) * ( 1 - exp_lambda(beta, x)**(-1) + 2 * alpha  * x * scale_pressure(beta, x))

#Q function
def Q_prime(alpha, beta, x):
    term1 = alpha * x * exp_lambda(beta, x) * (
        5 * scale_density(beta, x) +
        9 * scale_pressure(beta, x) +
        (scale_density(beta, x) + scale_pressure(beta, x)) / squared_sound_speed(beta, x)
    )
    term2 = 6 * exp_lambda(beta, x)
    term3 = (v_prime(alpha, beta, x)) ** 2
    return term1 - term2 - term3

# Radius that determine compactness for incompressible fluid of fixed mass
compactness = np.linspace(1e-3, 0.3, 5)
k_two = np.zeros(len(compactness))

for j, n in enumerate(compactness):
    alpha = (3 * n / 2) * ((2 - 3 * n) / (1 - 3 * n))

    def dydx(x, y):  # km^-1
        rhs = - (1 / 2 * x) * (y ** 2 + y * exp_lambda(n, x) *
                ( 1 + alpha * x * (scale_pressure(n, x) - scale_density(n, x))
                + Q_prime(alpha, n, x)))

        dy_dx = rhs / (2 * x)

        #print(f"metric values: e_lambda: {exp_lambda(n, x)}, Q: {Q_prime(alpha, n, x)}"
        #      f", nu_bar: {v_prime(alpha, n, x)}")
        #print(f"Analytic values: c^2: {squared_sound_speed(n, x)}, p/pc: {EOS_ratio(n, x)}.")
        return dy_dx

    # Surface discontinuity variable
    surface_density = ((1 - 2 * n) * (1 - 3 * n)) / ((1 - (3 * n / 2)) * (1 - n))

    y_sol = solve_ivp(dydx, (0.01, 1), [2], dense_output=True, method='Radau',
                            max_step = 1e-3, rtol=1e-11, atol=1e-13)

    #print(y_sol.y[0][-1])

    y_solutions = y_sol.y[0]
    y_R = y_solutions[-1] - (alpha / n) * (surface_density)

    print(y_solutions[-1])

    ""
    plt.figure()
    plt.plot(y_sol.t, y_sol.y[0])
    plt.xlabel('x')
    plt.ylabel('y', rotation = 0)
    plt.title(f'y vs. Radius for Incompressible Fluid with a mass of  {n:.4}')

    plt.show()
    ""

    def k2(beta, y_R):
        term1 = (8 / 5) * beta ** 5 * (1 - 2 * beta) ** 2 * (2 - y_R + 2 * beta * (y_R - 1))

        bracket1 = (
                2 * beta * (6 - 3 * y_R + 3 * beta * (5 * y_R - 8))
                + 4 * beta ** 3 * (13 - 11 * y_R + beta * (3 * y_R - 2) + 2 * beta ** 2 * (1 + y_R))
        )

        bracket2 = 3 * (1 - 2 * beta) ** 2 * (2 - y_R + 2 * beta * (y_R - 1)) * np.log(1 - 2 * beta)

        result = term1 / (bracket1 + bracket2)
        return result

    love_number = k2(compactness[j], y_R)

    print(f"The Love number for the Analytic Solution is {love_number:.3} "
          f"with a compactness of {compactness[j]:.3} where y_R = {y_R}.")

    k_two[j] = love_number

temp_df = pd.DataFrame({
        'compactness': compactness,
        'k2': k_two
    })
df = pd.concat([df, temp_df], ignore_index=True)
df.to_csv(output_filename, index=False)

fig, ax_k2 = plt.subplots(constrained_layout=True)
plt.title(r'$k_2$ for Analytic Solutions')
ax_k2.set_xlabel(r'$\beta$')
ax_k2.set_ylabel(r'$k_2$', rotation = 0)
ax_k2.set_xlim(0.0, 0.5)
ax_k2.set_ylim(0.0, 0.8)

ax_k2.plot(compactness, k_two)

plt.show()

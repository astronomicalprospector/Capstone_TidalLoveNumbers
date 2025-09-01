import os
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

output_filename = "AnalyticSolution_IncompressibleFluid.csv"

df = pd.DataFrame(columns=['compactness', 'k2'])
existing_indices = []

#pressure-uniform density ratio (Incompressible Fluid) using TOV equation
def EOS_ratio(beta, x):
    num = np.sqrt(1 - 2 * beta) - np.sqrt(1 - 2 * beta * x)
    den = np.sqrt(1 - 2 * beta * x) - 3 * np.sqrt(1 - 2 * beta)
    ratio = num / den
    return ratio

#metric functions (used in differentials)

def exp_lambda(beta, x): # dimensionless
    return (1 - 2 * beta * x)**(-1)

def v_prime(alpha, beta, x): # dimensionless
    return 2 * x * exp_lambda(beta, x) * ( beta + alpha * EOS_ratio(beta, x) )

#Q function
def Q_prime(alpha, beta, x): # dimensionless

    return ((alpha * x * exp_lambda(beta, x) * (5 + 9*EOS_ratio(beta, x))
            - 6 * exp_lambda(beta, x) - (v_prime(alpha, beta, x))**2))

# Radius that determine compactness for incompressible fluid of fixed mass
compactness = np.linspace(1e-3, 0.44, 100)
k_two = np.zeros(len(compactness))

for j, n in enumerate(compactness):

    alpha = 3 * n

    def dydx(x, y):  # km^-1
        rhs = -(y ** 2 + y + y * x * exp_lambda(n, x) *
                                              ( 2 * n + alpha * (EOS_ratio(n, x) - 1))
                                              + Q_prime(alpha, n, x))

        #print(f"metric values: e_lambda: {exp_lambda(n, x)}, Q: {Q_prime(alpha, n, x)}"
        #      f", nu_bar: {v_prime(alpha, n, x)}")
        #print(f"Term 1: {(1 / 2) *(y ** 2 + y)}, Term 2: {y * x * exp_lambda(n, x) *
        #                                      ( n + (alpha / 2) * (EOS_ratio(n, x) - 1))}, "
        #      f"Term 3: {Q_prime(alpha, n, x)}")
        #print(f"Analytic values: p/pc: {EOS_ratio(n, x)}.")
        dy_dx = rhs / (2 * x)

        return dy_dx

    y_sol = solve_ivp(dydx, (0.01, 1.0), [2], dense_output=True, method='Radau',
                            max_step = 1e-3, rtol=1e-11, atol=1e-13)
    y_R = y_sol.y[0][-1] - 3  #Correction for discontinuity

    print(y_sol.y[0][-1])

    ""
    plt.figure()
    plt.plot(y_sol.t, y_sol.y[0])
    plt.xlabel('x')
    plt.ylabel('y', rotation = 0)
    plt.title('y vs. Radius for Incompressible Fluid')

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

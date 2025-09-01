import os
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

output_filename = "AnalyticSolution_Buchdahl.csv"

df = pd.DataFrame(columns=['compactness', 'k2'])
existing_indices = []

#Coupled u and z variables
def scales(beta, x):
    def equations(p):
        u, z = p
        eq1 = u - (beta * np.sin(z) / z )
        eq2 = z - np.sqrt(x) * np.pi * ((1 - beta) / (1 - beta + u))
        return eq1, eq2

    initial_guess = (beta, np.pi * np.sqrt(x))
    u, z = fsolve(equations, initial_guess)
    return u, z

#Analytic Functions (Buchdahl Solution)
def scale_pressure(beta, u):
    p_num = beta * (1 - 2 * beta)
    p_den = (2 - 5 * beta) * ((1 - beta + u)**2)
    frac = p_num / p_den
    return frac * ((u / beta)**2)

def scale_density(beta, u):
    rho_num = (1 - 2 * beta) * (2 - 2 * beta - 3 * u)
    rho_den = (2 - 5 * beta) * ((1 - beta + u)**2)
    frac = rho_num / rho_den
    return frac * (u / beta)

def squared_sound_speed(beta, u):
    return u / (1 - beta - 4 * u)

#metric functions (used in differentials)
def exp_lambda(beta, u, z): # dimensionless
    e_num = (1 - 2 * beta) * (1 - beta + u)
    e_den = (1 - beta - u) * ((1 - beta + beta*np.cos(z))**2)
    return e_num / e_den

def v_prime(alpha, beta, x, u, z): # dimensionless
    return exp_lambda(beta, u, z) * (1 - (exp_lambda(beta, u, z))**(-1) + 2 * alpha  * x * scale_pressure(n, u))

#Q function
def Q_prime(alpha, beta, x, u, z): # dimensionless
    return ((alpha * x * exp_lambda(beta, u, z) * (5 * scale_density(n, u) + 9*scale_pressure(n, u) +
                                            (scale_density(n, u) + scale_pressure(n, u)) / (squared_sound_speed(beta, u)))
            - 6 * exp_lambda(beta, u, z) - (v_prime(alpha, beta, x, u, z))**2))

# Radius that determine compactness for incompressible fluid of fixed mass
compactness = np.linspace(1e-3, 0.45, 100)
k_two = np.zeros(len(compactness))

for j, n in enumerate(compactness):

    alpha = np.pi**2 * n * (1 - n)**2 * ((1 - 5 * n / 2) / (1 - 2 * n))

    def dydx(x, y):  # km^-1
        u, z = scales(n, x)

        rhs = -(y ** 2 + y * exp_lambda(n, u, z) *
                ( 1 + alpha * x * (scale_pressure(n, u) - scale_density(n, u)))
                  + Q_prime(alpha, n, x, u, z))

        dy_dx = rhs / (2 * x)

        #print(f"metric values: e_lambda: {exp_lambda(n, x)}, Q: {Q_prime(alpha, n, x)}"
        #      f", nu_bar: {v_prime(alpha, n, x)}")
        #print(f"Analytic values: c^2: {squared_sound_speed(n, x)}, p/pc: {EOS_ratio(n, x)}.")
        return dy_dx

    y_solutions = solve_ivp(dydx, (1e-4, 1), [2], dense_output=True, method='Radau',
                            max_step = 1e-3, rtol=1e-11, atol=1e-13)
    y_R = y_solutions.y[0][-1]

    """
    plt.figure()
    plt.plot(y_solutions.t, y_solutions.y[0])
    plt.xlabel('Radius (km)')
    plt.ylabel('y', rotation = 0)
    plt.title(f'y vs. Radius for Incompressible Fluid with a mass of  {n:.4}')

    plt.show()
    """

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

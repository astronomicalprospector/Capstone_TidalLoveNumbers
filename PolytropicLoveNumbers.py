import os
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from Poly_EOS import PolyEOS
from TOV import TOVSolver
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

output_filename = "love_numbers.csv"

df = pd.DataFrame(columns=['index', 'compactness', 'k2', 'mass', 'radius'])
existing_indices = []

#This script is more in line with the details seen in Tidal Love Number literature

#unit factor to include for energy density
r0 = 1.48  # km
p0 = 1.322e-6  # km^-2  # geometrized unit factor for pressure and energy density.
b = (4 * np.pi * p0 / r0) # km^-3

#metric functions (used in differentials)

def exp_lambda(m ,r): #unitless
    return (1 - (2 * r0 * m / r) )**(-1)

def nu_bar(m, p, r): # unitless
    return 2 * r0 * exp_lambda(m, r) * ((m / r) + b * p * r**2)

#Q function
def Q_prime(eps, m, p, r, c_s): # m^-2

    return ((b * r0 * r**2 * exp_lambda(m, r) * (5*eps + 9*p + (eps + p) / c_s))
            - 6 * exp_lambda(m, r)  - (nu_bar(m, p ,r))**2)

pc_length = 200

def compute_love_numbers(index):
    # Block that deals with EOS method,
    # Here we obtain the values of energy density that correspond to pressure and vice versa

    eos = PolyEOS(index)

    energy_density, pressure = eos.getEOS()

    tov = TOVSolver(eos)
    # This sub-block will utilize the TOV Solver in order to find MvR Curves for a set polytropic index
    p_c = np.linspace(0.1 , pressure[-1], pc_length)  # dimensionless

    compactness = np.zeros(len(p_c))
    k_two = np.zeros(len(p_c))
    mass = np.zeros(len(p_c))
    radius = np.zeros(len(p_c))

    for j, n in enumerate(p_c):
        p_tov, m_tov, r_tov = tov.solve_tov(n)

        compactness[j] = r0 * (m_tov[-1] / r_tov[-1])
        mass[j] = m_tov[-1]
        radius[j] = r_tov[-1]

        print(f"Characteristic Mass and Radius of NS with p_c = {n} is {m_tov[-1]} Solar Masses, "
              f"{r_tov[-1]} km.")

        # Interpolations
        mass_interp = CubicSpline(r_tov, m_tov)
        pressure_interp = CubicSpline(r_tov, p_tov)

        def cs_squared(p, e):
            return (p / e) * (1 + 1 / index)

        # tidal love number differential equation

        def dydr(r, y):  # km^-1
            p = pressure_interp(r)
            ed = eos.findED(pressure_interp(r))
            m = mass_interp(r)

            dy_dr = - (1 / r) * (y ** 2 + y + r0 * y * exp_lambda(m, r) * ((2 * m / r) + b * r ** 2 * (p - ed))
                                 + Q_prime(ed, m, p, r, cs_squared(p, ed)))

            #print(f"metric values: e_lambda: {exp_lambda(m, r)}, Q: {Q_prime(ed, m, p, r, cs_squared(p, ed))}"
            #      f", nu_bar: {nu_bar(m, p, r)}")
            #print(f"parameters: pressure: {p}, energy density: {ed}, mass {m}, radius {r},"
            #      f" sound speed: {cs_squared(p, ed)}")

            return dy_dr

        def stop_condition(r, y):
            return pressure_interp(r)

        stop_condition.terminal = True
        stop_condition.direction = -1


        tolerance = 1e-9 if index < 0.6 else 1e-8
        y_solutions = solve_ivp(dydr, (r_tov[0], r_tov[-1]), [2], dense_output=True,
                                    method='Radau', rtol = tolerance, atol = 1e-14, events = stop_condition,
                                max_step = 1e-3)
        y_R = y_solutions.y[0][-1]

        """
        plt.figure()
        plt.plot(y_solutions.t, y_solutions.y[0])
        plt.xlabel('Radius (km)')
        plt.ylabel('y')
        plt.title(f'y vs. Radius for index n = {index:.4} with central pressure {n:.4}')

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

        print(f"The Love number for the given EOS Model is {love_number:.3} "
              f"with a compactness of {compactness[j]:.3} where y_R = {y_R}")

        k_two[j] = love_number

    """
    plt.figure()
    plt.plot(compactness, k_two)
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$k_2$')
    plt.title(f'Tidal Love Number for index n = {index:.4}')
    plt.show()
    """

    return compactness, k_two, mass, radius

n_values = np.arange(0.2, 2.1, 0.1).round(3).tolist()

tidal_matrix = np.zeros((len(n_values), pc_length))
compactness_matrix = np.zeros((len(n_values), pc_length))

norm = mcolors.Normalize(vmin=min(n_values), vmax=max(n_values))
cmap = plt.colormaps['coolwarm']

""
fig, ax_k2 = plt.subplots(constrained_layout=True)
plt.title(r'$k_2$ for Various Polytropic Equations of State')
ax_k2.set_xlabel(r'$\beta$')
ax_k2.set_ylabel(r'$k_2$')
ax_k2.set_xlim(0.0, 0.35)
ax_k2.set_ylim(0.0, 0.5)
""

for j, n in enumerate(n_values):
    Beta, k_2, m, r = compute_love_numbers(n)
    print(f"Completed computation of tidal curve for index {n}.")

    temp_df = pd.DataFrame({
        'index': [n] * pc_length,
        'compactness': Beta,
        'k2': k_2,
        'mass': m,
        'radius': r
    })

    df = pd.concat([df, temp_df], ignore_index=True)
    df.to_csv(output_filename, index=False)

    if np.isclose(n, 1.0):
        ax_k2.plot(Beta, k_2, color='black', linewidth=1.5, label='n = 1')
    else:
        ax_k2.plot(Beta, k_2, color=cmap(norm(n)), linewidth=1)

plt.show()


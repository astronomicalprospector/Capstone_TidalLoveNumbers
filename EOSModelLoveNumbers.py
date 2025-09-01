import os
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from Model_TOV import TOVSolver
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

from scipy.interpolate import interp1d


output_filename = "Plots/RMF4_low_pressure.csv"

df = pd.DataFrame(columns=['compactness', 'k2', 'Solar Masses', 'Radius (km)'])
existing_indices = []

#geometrized units
r0 = 1.48  # km
e0_geo = 0.003006 * r0  # geometrized unit factor for pressure and energy density. (km^-2)

b = (4 * np.pi * e0_geo / r0) # km^-3

#metric functions (used in differentials)

def exp_lambda(m ,r): #unitless
    return (1 - (2 * r0 * m / r) )**(-1)

def nu_bar(m, p, r): # unitless
    return 2 * r0 * exp_lambda(m, r) * ((m / r) + b * p * r**2)

#Q function
def Q_prime(eps, m, p, r, c_s): # unitless

    return ((b * r0 * r**2 * exp_lambda(m, r) * (5*eps + 9*p + (eps + p) / c_s))
            - 6 * exp_lambda(m, r)  - (nu_bar(m, p ,r))**2)

EOS_data = np.loadtxt('EOS Models/RMF/RMF4_eos.thermo', skiprows=1)
nb_data = np.loadtxt('EOS Models/RMF/RMF4_eos.nb', skiprows=2)

scaled_pressure = EOS_data[:, 3] # MeV, this is pressure scaled with number density (p / n_b)
e_internal = EOS_data[:, 9]  # Dimensionless, this is the scaled internal energy per baryon [(e/n_b m_n) - 1]

m_n = 939.565  # MeV

pressure = scaled_pressure * nb_data  # MeV/fm^3
energy_density = (e_internal + 1) * m_n * nb_data  # MeV/fm^3

#Natural Unit to Geometrized Unit Conversion
#This uses the fiducial pressure term p0 introduced in section IV
#There is then another factor in place which then makes them dimensionless
p0 = 1.322e-6  # km^-2 MeV^-1 fm^3
pressure = pressure * p0 / e0_geo
energy_density = energy_density * p0 / e0_geo

"""
plt.plot(energy_density, pressure, 'black')
plt.xlabel('ϵ')
plt.ylabel('p', rotation=0)
plt.title('Pressure vs. Energy Density')

plt.show()
"""

tov = TOVSolver(nb_data, energy_density, pressure)

p_c = np.linspace(1e-4, pressure[-1], 600) # dimensionless
mass = np.zeros(len(p_c))  # dimensionless
radii = np.zeros(len(p_c))  # km

#print(p_c)

compactness = np.zeros(len(p_c))
k_two = np.zeros(len(p_c))

# My clever attempt at getting energy density from pressure despite it not strictly increasing
# This interpolation scheme finds the baryon number density according to a given pressure
# Then it uses that value of number density to find its respective energy density
# This still technically finds energy density from a given pressure with an extra step involving
# The number density at that pressure point
EOS_internal_interpolator = interp1d(pressure, nb_data, kind ='linear', fill_value= "extrapolate")
EOS_interpolator_type2 = interp1d(nb_data, energy_density, kind ='linear', fill_value= "extrapolate")
def findED(p):
    return EOS_interpolator_type2(EOS_internal_interpolator(p))

for j, n in enumerate(p_c):
    p_tov, m_tov, r_tov = tov.solve_tov(n)
    mass[j] = m_tov[-1]
    radii[j] = r_tov[-1]

    compactness[j] = r0 * (m_tov[-1] / r_tov[-1])

    print(f"Characteristic Mass and Radius of NS with p_c = {n} is {m_tov[-1]} Solar Masses, "
          f"{r_tov[-1]} km.")

    # Interpolations
    mass_interp = CubicSpline(r_tov, m_tov)
    pressure_interp = CubicSpline(r_tov, p_tov)

    # Sound Speed Interpolator
    cs2_interpolator = CubicSpline(energy_density, np.gradient(pressure, energy_density))

    # tidal love number differential equation

    def dydr(r, y):  # km^-1
        p = pressure_interp(r)

        ed = findED(p)
        m = mass_interp(r)

        cs2 = cs2_interpolator(ed)

        if p < pressure[0]:
            return [0]

        #print(p, ed, cs2, r)

        dy_dr = - (1 / r) * (y ** 2 + y + r0 * y * exp_lambda(m, r) * ((2 * m / r) + b * r ** 2 * (p - ed))
                             + Q_prime(ed, m, p, r, cs2))

        #print(f"metric values: e_lambda: {exp_lambda(m, r)}, Q: {Q_prime(ed, m, p, r, cs2)}"
        #      f", nu_bar: {nu_bar(m, p, r)}")

        return dy_dr

    def stop_condition(r, y):
        return pressure_interp(r) - pressure[0]

    stop_condition.terminal = True
    stop_condition.direction = -1


    y_solutions = solve_ivp(dydr, (r_tov[0], r_tov[-1]), [2], dense_output=True,
                                method='Radau', rtol=1e-10, atol=1e-12, events=stop_condition)

    y_R = y_solutions.y[0][-1]

    if not y_solutions.success:
        print(y_solutions.y[0][-1], y_solutions.t[-1])
        print("Integration failed:", y_solutions.message)

    ""
    plt.figure()
    plt.plot(y_solutions.t, y_solutions.y[0])
    plt.xlabel('Radius (km)')
    plt.ylabel('y')
    plt.title(f'y vs. Radius with central pressure {n:.4}')

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

temp_df = pd.DataFrame({
    'k2': k_two,
    'compactness': compactness,
    'Solar Masses': mass,
    'Radius (km)': radii,
})

df = pd.concat([df, temp_df], ignore_index=True)
df.to_csv(output_filename, index=False)

plt.plot(radii, mass, 'black')
plt.xlabel('R (km)')
plt.ylabel(r'M ($M_\odot$)', rotation=0)
plt.title('Mass vs. Radius Curve')

plt.show()

fig, ax_k2 = plt.subplots(constrained_layout=True)
plt.title(r'$k_2$ for Equation of State')
ax_k2.set_xlabel(r'$\beta$')
ax_k2.set_ylabel(r'$k_2$')

ax_k2.plot(compactness, k_two, color='black')

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.optimize import fsolve

#MeV (masses for all quark flavors and leptons (electrons and muons)
M_U = 5
M_D = 7
M_S = 150
M_C = 1500

#Bag Constant MeV^(1/4)
B = 154.5

def pressure_of_flavor(mu_f, k_f, m_f):
    term1 = mu_f * k_f * (mu_f**2 - (5 / 2) * m_f**2)
    term2 = (3 / 2) * m_f**4 * np.log((mu_f + k_f) / m_f)
    return (4 * np.pi**2)**(-1) * (term1 + term2)

def energy_density_of_flavor(mu_f, k_f, m_f):
    term1 = mu_f * k_f * (mu_f**2 - (1 / 2) * m_f**2)
    term2 = (1 / 2) * m_f**4 * np.log((mu_f + k_f) / m_f)
    return (3 / (4 * np.pi**2)) * (term1 - term2)


def functions(chem_potentials):
    mu_u, mu_e = chem_potentials

    charge_neutrality_condition = (2 * (mu_u**2 - M_U**2)**(3/2) - (mu_u**2 + (2 * mu_u * mu_e) + mu_e**2 - M_D**2)**(3/2)
                - (mu_u**2 + 2 * mu_u * mu_e + mu_e**2 - M_S**2)**(3/2) - mu_e**3)
    conserved_baryon_density_condition =
    return charge_neutrality_condition, conserved_baryon_density_condition

mu_u_guess, mu_e_guess = 1.5*B, (1/2)*B

chemical_potentials = fsolve(functions, [mu_u_guess, mu_e_guess])

print(chemical_potentials)



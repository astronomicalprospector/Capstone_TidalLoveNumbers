import numpy as np
import matplotlib.pyplot as plt
from NSM_EOS import EOSConstruct
from TOV import TOVSolver
from scipy.interpolate import PchipInterpolator
from scipy.integrate import odeint

#This script utilizes an EOS constructed from an array of number densities sourced from CompOSE

#unit factor to include for energy density
r0 = 1.48  # km
rho_0 = 1.249e-4  # km^-2  # geometrized unit factor for pressure and energy density. (km^-2)

b = (4 * np.pi * rho_0 / r0) # km^-3

#metric functions (used in differentials)

def exp_lambda(m ,r): #unitless
    return (1 - (2 * r0 * m / r) )**(-1)

def nu(m, p, r): # m^-1
    return 2 * r0 * exp_lambda(m, r) * (m + b * p * r**3) / r**2

#Q function
def Q(p, r): # m^-2
    c_s = sound_speed_interp(r)
    m = mass_interp(r)
    eps = eos.getED(p)

    return ((b * r0 * exp_lambda(m, r) * (5*eps + 9*p + (eps + p)/c_s))
            - 6 * (exp_lambda(m, r) / r**2)  - nu(m, p ,r)**2)


#tidal love number differential equation

def dydr(r, y): # m^-1
    p = pressure_interp(r)
    eps = eos.getED(p)
    m = mass_interp(r)

    dy_dr = - (1 / r) * (y**2 + y * exp_lambda(m, r) *
                         (1 + 4 * np.pi * r**2 * (p - eps)) + r**2 * Q(p, r))
    return dy_dr

#Block that deals with EOS method,
#Here we obtain the values of energy density that correspond to pressure and vice versa
with open(f'CNSMeos.nb') as number_density:
    input_data = number_density.readlines()

baryon_number_density = input_data[2:]
baryon_number_density = np.array(baryon_number_density, dtype=float)


eos = EOSConstruct("double_poly")

pressure, energy_density = eos.construct_eosfg(baryon_number_density)
plt.plot(energy_density, pressure, 'black')
plt.xlabel('ϵ')
plt.ylabel('p')
plt.title('Energy Density vs. Pressure')

#Block of code that extracts mass and pressure as functions of current radius,
#Will include an interpolation function for the mass so that it is known for a given radius

central_pressure = 1e-3  #dimensionless
tov = TOVSolver(eos)

p_tov, m_tov, r_tov = tov.solve_tov(central_pressure)

plt.figure()
plt.plot(r_tov, m_tov, 'black')
plt.xlabel('Radius (km)')
plt.ylabel(r'Mass (M $\odot$)')
plt.title('Mass vs. Radius')

plt.figure()
plt.plot(r_tov, p_tov, 'black')
plt.xlabel('Radius (km)')
plt.ylabel(r'Pressure (km^-2)')
plt.title('Pressure vs. Radius')

plt.show()

#Block of code that will use the all the given variables to then find tidal love numbers for a given compactness

#Defining Interpolators
mass_interp = PchipInterpolator(r_tov, m_tov)
pressure_interp = PchipInterpolator(r_tov, p_tov)

sound_speed = np.gradient(pressure, energy_density)
r_range = np.linspace(r_tov[0], r_tov[-1], len(sound_speed))
sound_speed_interp = PchipInterpolator(r_range, sound_speed)

beta = 1.48 * m_tov[-1] / r_tov[-1]

print(beta)

y_0 = 2
sol = odeint(dydr, y0 = y_0, t=r_tov, tfirst = True)

y_R = sol[-1]

k_two = ((8/5) * beta** 5 * (1 - 2 *beta)**2 * (2 - y_R + 2 * beta * (y_R - 1)) *
         (2 * beta * (6 - 3 * y_R + 3 * beta * (5 * y_R - 8) + 2 * beta**2 * (13 - 11 * y_R + beta*(3 * y_R - 2) +
                                                                           2 * beta**2 * (1 + y_R))) +
          3 * (1 - 2 * beta)**2 * (2 - y_R + 2 * beta * (y_R - 1))) * np.log(1 - 2 * beta)**-1)

print(f"The Love number for the given EOS Model is {k_two}")

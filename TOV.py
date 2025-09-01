import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class TOVSolver:

    # Differentiation interval setup
    dr = 1e-3
    r_max = 50

    r0 = 1.48 # km
    p0 = 1.322e-6  # km^-2  # geometrized unit factor for pressure and energy density.

    b = (4 * np.pi * p0 / r0) # km^-3

    def __init__(self, eos):

        self.eos = eos

    def solve_tov(self, central_pressure):
        def dtdr(r, T):
            p, m = T
            ed = self.eos.findED(p)

            dp_dr = - self.r0 * ((ed + p) * (m + self.b * r**3 * p))/ (r * (r - 2 * self.r0 * m)) # km^-1

            dm_dr = self.b * ed * r**2 #km^-1


            #print(f"r: {r}, p: {p}, m: {m}, ed: {ed}, dp/dr: {dp_dr}, dm/dr: {dm_dr}")

            return [dp_dr, dm_dr]


        def stop_condition(r, T):
            p, _ = T
            return p

        stop_condition.terminal = True
        stop_condition.direction = -1

        m_init = 0

        init_cond = (central_pressure, m_init)

        tov_solutions = solve_ivp(dtdr, (1e-6, self.r_max), init_cond,
                   events = stop_condition, dense_output=True,
                                  method='Radau', rtol=1e-13, atol=1e-15)

        pressure = tov_solutions.y[0]                 #no dimensions (multiply by e0_geo to get km^-2 units)
        mass = tov_solutions.y[1]                     #M_sol
        radii = tov_solutions.t                       #km

        """
        fig, ax1 = plt.subplots()
        plt.title(f'Pressure and Mass vs Radius')

            #Pressure Plotting
        color1 = 'tab:blue'
        ax1.set_xlabel('Radius (km)')
        ax1.set_ylabel('Pressure', color=color1)
        ax1.plot(radii, pressure, color=color1, label='Pressure')
        ax1.tick_params(axis='y', labelcolor=color1)

            #Mass Plotting
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Mass ($M_\\odot$)', color=color2)
        ax2.plot(radii, mass, color=color2, label='Mass')
        ax2.tick_params(axis='y', labelcolor=color2)

        plt.show()
        """

        return pressure, mass, radii
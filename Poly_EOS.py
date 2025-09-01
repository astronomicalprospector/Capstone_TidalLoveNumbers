import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


class PolyEOS:
    #fiducial terms of dimension km^-2
    p0 = 1.322e-6
    e0 = 1.249e-4

    def __init__(self, n):
        self.n = n

        self.log_k = np.log(self.p0) - (1 + (1 / n)) * np.log(self.e0)
        self.log_k_nodim = self.log_k + (1/n) * np.log(self.p0)
        #print(self.log_k_nodim)

        self.energy_density = np.logspace(-6, 6, 10000000)
        self.pressure = np.zeros(len(self.energy_density))

        self.log_pressure = np.zeros(len(self.energy_density))
        self.log_energy_density = np.log(self.energy_density)

        for j, ed in enumerate(self.log_energy_density):

            self.log_pressure[j] = self.log_k_nodim + (1 + 1 /n) * ed

        p = np.exp(self.log_pressure)
        rho = np.exp(self.log_energy_density)

        sound_speed = np.gradient(p, rho)
        # Create mask so EOS does not violate causality
        mask = np.abs(sound_speed) < 1

        self.pressure = p[mask]
        self.energy_density = rho[mask]

        """
        fig, ax_eos = plt.subplots(constrained_layout=True)
        plt.title(f'Pressure vs. Energy Density for index n = {n:.4}')
        ax_eos.set_xlabel('ϵ')
        ax_eos.set_ylabel('p')
        ax_eos.plot(self.energy_density , self.pressure , 'black')

        plt.show()
        """

    def findED(self, p_i):
        pressure = np.maximum(p_i, 1e-25)
        log_pressure = np.log(pressure)
        log_rho = (log_pressure - self.log_k_nodim) * ((1 / (1 + 1/self.n)))
        rho = np.exp(log_rho)

        return (rho)

    def getEOS(self):
        return self.energy_density, self.pressure


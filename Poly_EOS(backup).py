import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt


class PolyEOS:
    #fiducial terms of dimension km^-2
    p0 = 1.322e-6
    e0 = 1.249e-4

    def __init__(self, n):
        self.n = n

        self.log_k = np.log(self.p0) - (1 + (1 / n)) * np.log(self.e0)
        print(self.log_k)
        self.k = np.exp(self.log_k) * (self.p0 ** (1 / n))

        print(self.k)

        self.energy_density = np.logspace(-6, 3, 10000)
        self.pressure = np.zeros(len(self.energy_density))

        for j, ed in enumerate(self.energy_density):

            self.pressure[j] = self.k * ed**(1 + 1 / n)

        sound_speed = np.gradient(self.pressure, self.energy_density)
        # Create mask so EOS does not violate causality
        mask = np.abs(sound_speed) < 1

        self.pressure = self.pressure[mask]
        self.energy_density = self.energy_density[mask]

        ""
        fig, ax_eos = plt.subplots(constrained_layout=True)
        plt.title(f'Pressure vs. Energy Density for index n = {n:.4}')
        ax_eos.set_xlabel('ϵ')
        ax_eos.set_ylabel('p')
        ax_eos.plot(self.energy_density , self.pressure , 'black')

        plt.show()
        ""

    def findED(self, p_i):
        #pressure = np.maximum(p_i, 1e-18)
        rho = (p_i / self.k) ** (self.n / (self.n + 1))

        return (rho)

    def getEOS(self):
        return self.energy_density, self.pressure


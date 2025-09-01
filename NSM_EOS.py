import numpy as np
from scipy.optimize import curve_fit

class EOSConstruct:

    #Polytropic coefficients (initialized)
    k = [0, 0, 0]
    n = [0, 0, 0]

    # MeV (for all fermion masses)
    M_E = 0.511
    M_N = 939.5653
    M_P = 938.2720

    def __init__(self, polytype):
        # dimension factor
        self.e_0 = self.M_N**4 / ((3*np.pi**2)**3) # MeV^4

        self.polytype = polytype

    def construct_eosfg(self, n_b):
        nb = np.array(n_b)

        # energy density and pressure arrays
        self.energy_density = np.zeros(len(nb))
        self.pressure = np.zeros(len(nb))

        # fermi momentum of neutron based on nb
        k_fn = (3*np.pi**2 * nb)**(1/3) * 197.33  # MeV

        # proton fermi momentum determined by weak interaction equilibrium or beta equilibrium
        k_fp = (np.sqrt((k_fn ** 2 + self.M_N ** 2 - self.M_E ** 2) ** 2 - 2 * self.M_P ** 2 *
                        (k_fn ** 2 + self.M_N ** 2 + self.M_E ** 2) + self.M_P ** 4)
                        / (2 * np.sqrt(k_fn ** 2 + self.M_N ** 2)))

        # Charge neutrality makes any instance of electron fermi momentum equal to proton fermi momentum

        # setting parameters
        # All x and y parameters are dimensionless quantities
        x_p = k_fp / self.M_P
        x_n = k_fn / self.M_N
        x_e = k_fp / self.M_E

        # Energy Density and Pressure solving

        def pressure_func(x):
            return x * (2 * x ** 2 + 1) * np.sqrt(1 + x ** 2) - np.arcsinh(x)

        def edensity_func(x):
            return x * (2 * x ** 2 - 3) * np.sqrt(1 + x ** 2) + 3 * np.arcsinh(x)

        # stacking parameter x to loop over for each fermion
        x = np.vstack([x_p, x_n, x_e])

        n = 0
        while n < 2:
            param = x[n, :]
            p = np.zeros(len(x_n))
            ed = np.zeros(len(x_n))

            for j, i in enumerate(param):
                p[j] = pressure_func(i)
                ed[j] = edensity_func(i)

            self.energy_density = self.energy_density + ed
            self.pressure = self.pressure + p
            n += 1

        # Dimensionless
        energy_density = self.energy_density
        pressure = self.pressure

        if self.polytype == 'single_poly':
            EOSConstruct.polytrope(self, energy_density, pressure)
        else:
            EOSConstruct.db_polytrope(self, energy_density, pressure)

        return energy_density, pressure

    def polytrope(self, p, e):
        def poly_f(e, k, n):
            return k * e**(1 + 1/n)

        popt, pcov = curve_fit(poly_f, e, p, p0=[1, 3 / 5])
        self.k[0], self.n[0], = popt
        print(popt)

        return popt

    def db_polytrope(self, p, e):
        def db_poly_f(p, a1, a2, g1, g2):
            return (a1 * p ** g1) + (a2 * p ** g2)

        popt, pcov = curve_fit(db_poly_f, p, e, p0=[2, 3, 3 / 5, 1])
        self.k[1], self.k[2], self.n[1], self.n[2], = popt

        print(popt)

        return popt


    def getED(self, p_i):
        if self.polytype == 'single_poly':
            eps = (p_i / self.k[0]) ** (self.n[0] / (self.n[0] + 1))
            return eps
        if self.polytype == 'double_poly':
            eps = self.k[1] * p_i ** self.n[1] + self.k[2] * p_i ** self.n[2]
            return eps
        else:
            return "Invalid polytrope input. Options are single_poly or double_poly."


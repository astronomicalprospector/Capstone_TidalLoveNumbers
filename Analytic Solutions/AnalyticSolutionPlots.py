import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors

Incompressible = pd.read_csv('AnalyticSolution_IncompressibleFluid.csv')
TolmanVII = pd.read_csv('AnalyticSolution_TolmanVII.csv')
Buchdahl = pd.read_csv('AnalyticSolution_Buchdahl.csv')

norm = mcolors.Normalize(vmin=1, vmax=6)
cmap = plt.colormaps['inferno']

fig, ax_k2 = plt.subplots(constrained_layout=True)
ax_k2.set_xlabel(r'$\beta$')
ax_k2.set_ylabel(r'$k_2$', rotation =  0)

ax_k2.plot(Incompressible['compactness'], Incompressible['k2'], color= cmap(norm(1)), label='Incompressible Fluid')
ax_k2.plot(TolmanVII['compactness'], TolmanVII['k2'], color= cmap(norm(2)), label='Tolman VII')
ax_k2.plot(Buchdahl['compactness'], Buchdahl['k2'], color= cmap(norm(3)), label='Buchdahl')

ax_k2.legend()

plt.show()
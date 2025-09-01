import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors

SQM_data = pd.read_csv('SQM_Massless_approximation.csv')

QHC19A_data = pd.read_csv('QHC19-A.csv')

polarizability = np.log10((2/3) * (SQM_data['k2'] * (SQM_data['Radius (km)'])**5))

polarizability_QHC19A = np.log10((2/3) * (QHC19A_data['k2'] * (QHC19A_data['Radius (km)'])**5))


norm = mcolors.Normalize(vmin=1, vmax=5)
cmap = plt.colormaps['inferno']

fig, ax_k2 = plt.subplots(constrained_layout=True)
ax_k2.set_xlabel(r'$\beta$')
ax_k2.set_ylabel(r'$k_2$', rotation =  0)
ax_k2.set_ylim(0, 0.8)
ax_k2.set_xlim(0, 0.35)

ax_k2.plot(SQM_data['compactness'], SQM_data['k2'], color= cmap(norm(1)))
ax_k2.scatter(SQM_data['compactness'][43], SQM_data['k2'][43], marker = "o", s = 12, color= 'black')
ax_k2.scatter(SQM_data['compactness'][23], SQM_data['k2'][23], facecolors='none', edgecolors='b', s = 17)

fig, ax_mvr = plt.subplots(constrained_layout=True)
ax_mvr.set_xlabel('Radius (km)')
ax_mvr.set_ylabel(r'M ($M_\odot$)', rotation = 0)
ax_mvr.set_xlim(4, 12)

ax_mvr.plot(SQM_data['Radius (km)'], SQM_data['Solar Masses'], color= cmap(norm(1)))
ax_mvr.scatter(SQM_data['Radius (km)'][43], SQM_data['Solar Masses'][43], marker = "o", s = 12, color= 'black')
ax_mvr.scatter(SQM_data['Radius (km)'][23], SQM_data['Solar Masses'][23], facecolors='none', edgecolors='b', s = 17)

fig, ax_lambda = plt.subplots(constrained_layout=True)
ax_lambda.set_xlabel(r'$R (km)$')
ax_lambda.set_ylabel(r'$log_{10}\lambda (km^{5})$')
ax_lambda.set_xlim(6, 15)
ax_lambda.set_ylim(3.3, 5)

ax_lambda.plot(SQM_data['Radius (km)'], polarizability, color= cmap(norm(1)))
ax_lambda.scatter(SQM_data['Radius (km)'][43], polarizability[43], marker = "o", s = 12, color= 'black')
ax_lambda.scatter(SQM_data['Radius (km)'][23], polarizability[23], facecolors='none', edgecolors='b', s = 17)

ax_lambda.plot(QHC19A_data['Radius (km)'], polarizability_QHC19A, color= cmap(norm(2)))
ax_lambda.scatter(QHC19A_data['Radius (km)'][23], polarizability_QHC19A[23], marker = "o", s = 10, color= 'black')
ax_lambda.scatter(QHC19A_data['Radius (km)'][11], polarizability_QHC19A[11], facecolors='none', edgecolors='b', s = 15)
ax_lambda.annotate('QHC19A', xy=(QHC19A_data['Radius (km)'][100], polarizability_QHC19A[100]),
               xytext=(QHC19A_data['Radius (km)'][100]+0.01, polarizability_QHC19A[100]+0.01),
               color= cmap(norm(2)))

plt.show()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors

RMF1_data = pd.read_csv('RMF1.csv')
RMF2_data = pd.read_csv('RMF2.csv')
RMF3_data = pd.read_csv('RMF3.csv')
SLY4_data = pd.read_csv('SLY4.csv')
GPPVA_data = pd.read_csv('GPPVA.csv')
QHC19A_data = pd.read_csv('QHC19-A.csv')

#quadrupole polarizabilities (logarithmic)
polarizability_RMF1 = np.log10((2/3) * (RMF1_data['k2'] * (RMF1_data['Radius (km)'])**5))
polarizability_RMF2 = np.log10((2/3) * (RMF2_data['k2'] * (RMF2_data['Radius (km)'])**5))
polarizability_RMF3 = np.log10((2/3) * (RMF3_data['k2'] * (RMF3_data['Radius (km)'])**5))
polarizability_SLY4 = np.log10((2/3) * (SLY4_data['k2'] * (SLY4_data['Radius (km)'])**5))
polarizability_GPPVA = np.log10((2/3) * (GPPVA_data['k2'] * (GPPVA_data['Radius (km)'])**5))
polarizability_QHC19A = np.log10((2/3) * (QHC19A_data['k2'] * (QHC19A_data['Radius (km)'])**5))


norm = mcolors.Normalize(vmin=1, vmax=10)
cmap = plt.colormaps['inferno']

fig, ax_k2 = plt.subplots(constrained_layout=True)
ax_k2.set_xlabel(r'$\beta$')
ax_k2.set_ylabel(r'$k_2$', rotation =  0)
ax_k2.set_ylim(0, 0.2)

ax_k2.plot(RMF1_data['compactness'], RMF1_data['k2'], color= cmap(norm(1)))
ax_k2.plot(RMF2_data['compactness'], RMF2_data['k2'], color= cmap(norm(2)))
ax_k2.plot(RMF3_data['compactness'], RMF3_data['k2'], color= cmap(norm(3)))
ax_k2.plot(SLY4_data['compactness'], SLY4_data['k2'], color= cmap(norm(4)))
ax_k2.plot(GPPVA_data['compactness'], GPPVA_data['k2'], color= cmap(norm(5)))
ax_k2.plot(QHC19A_data['compactness'], QHC19A_data['k2'], color= cmap(norm(6)))

ax_k2.scatter(RMF1_data['compactness'][62], RMF1_data['k2'][62], marker = "o", s = 10, color= 'black')
ax_k2.scatter(RMF1_data['compactness'][28], RMF1_data['k2'][28], facecolors='none', edgecolors='b', s = 15)
ax_k2.annotate('RMF1', xy=(RMF1_data['compactness'][100], RMF1_data['k2'][100]),
               xytext=(RMF1_data['compactness'][100]+0.01, RMF1_data['k2'][100]+0.01),
               arrowprops=dict(arrowstyle="-"), color= cmap(norm(1)))

ax_k2.scatter(RMF2_data['compactness'][46], RMF2_data['k2'][46], marker = "o", s = 10, color= 'black')
ax_k2.scatter(RMF2_data['compactness'][22], RMF2_data['k2'][22], facecolors='none', edgecolors='b', s = 15)
ax_k2.annotate('RMF2', xy=(RMF2_data['compactness'][100], RMF2_data['k2'][100]),
               xytext=(RMF2_data['compactness'][100]+0.01, RMF2_data['k2'][100]+0.01),
               arrowprops=dict(arrowstyle="-"), color= cmap(norm(2)))

ax_k2.scatter(RMF3_data['compactness'][35], RMF3_data['k2'][35], marker = "o", s = 10, color= 'black')
ax_k2.scatter(RMF3_data['compactness'][17], RMF3_data['k2'][17], facecolors='none', edgecolors='b', s = 15)
ax_k2.annotate('RMF3', xy=(RMF3_data['compactness'][100], RMF3_data['k2'][100]),
               xytext=(RMF3_data['compactness'][100]+0.01, RMF3_data['k2'][100]+0.01),
               arrowprops=dict(arrowstyle="-"), color= cmap(norm(3)))

ax_k2.scatter(SLY4_data['compactness'][18], SLY4_data['k2'][18], marker = "o", s = 10, color= 'black')
ax_k2.scatter(SLY4_data['compactness'][8], SLY4_data['k2'][8], facecolors='none', edgecolors='b', s = 15)
ax_k2.annotate('SLY4', xy=(SLY4_data['compactness'][100], SLY4_data['k2'][100]),
               xytext=(SLY4_data['compactness'][100]+0.01, SLY4_data['k2'][100]+0.01),
               arrowprops=dict(arrowstyle="-"), color= cmap(norm(4)))

ax_k2.scatter(GPPVA_data['compactness'][13], GPPVA_data['k2'][13], marker = "o", s = 10, color= 'black')
ax_k2.scatter(GPPVA_data['compactness'][7], GPPVA_data['k2'][7], facecolors='none', edgecolors='b', s = 15)
ax_k2.annotate('GPPVA', xy=(GPPVA_data['compactness'][120], GPPVA_data['k2'][120]),
               xytext=(GPPVA_data['compactness'][120]+0.01, GPPVA_data['k2'][120]+0.01),
               arrowprops=dict(arrowstyle="-"), color= cmap(norm(5)))

ax_k2.scatter(QHC19A_data['compactness'][23], QHC19A_data['k2'][23], marker = "o", s = 10, color= 'black')
ax_k2.scatter(QHC19A_data['compactness'][11], QHC19A_data['k2'][11], facecolors='none', edgecolors='b', s = 15)
ax_k2.annotate('QHC19-A', xy=(QHC19A_data['compactness'][100], QHC19A_data['k2'][100]),
               xytext=(QHC19A_data['compactness'][100]+0.01, QHC19A_data['k2'][100]+0.01),
               arrowprops=dict(arrowstyle="-"), color= cmap(norm(6)))

plt.show()

fig, ax_mvr = plt.subplots(constrained_layout=True)
ax_mvr.set_xlabel('Radius (km)')
ax_mvr.set_ylabel(r'M ($M_\odot$)', rotation = 0)
ax_mvr.set_xlim(8, 15)

ax_mvr.plot(RMF1_data['Radius (km)'], RMF1_data['Solar Masses'], color= cmap(norm(1)))
ax_mvr.plot(RMF2_data['Radius (km)'], RMF2_data['Solar Masses'], color= cmap(norm(2)))
ax_mvr.plot(RMF3_data['Radius (km)'], RMF3_data['Solar Masses'], color= cmap(norm(3)))
ax_mvr.plot(SLY4_data['Radius (km)'], SLY4_data['Solar Masses'], color= cmap(norm(4)))
ax_mvr.plot(GPPVA_data['Radius (km)'], GPPVA_data['Solar Masses'], color= cmap(norm(5)))
ax_mvr.plot(QHC19A_data['Radius (km)'], QHC19A_data['Solar Masses'], color= cmap(norm(6)))


ax_mvr.annotate('RMF1', xy=(RMF1_data['Radius (km)'][10], RMF1_data['Solar Masses'][10]),
                xytext=(RMF1_data['Radius (km)'][10]+0.5, RMF1_data['Solar Masses'][10]+0.1),
                arrowprops=dict(arrowstyle="-"), color= cmap(norm(1)))

ax_mvr.annotate('RMF2', xy=(RMF2_data['Radius (km)'][10], RMF2_data['Solar Masses'][10]),
                xytext=(RMF2_data['Radius (km)'][10]+0.5, RMF2_data['Solar Masses'][10]+0.1),
                arrowprops=dict(arrowstyle="-"), color= cmap(norm(2)))

ax_mvr.annotate('RMF3', xy=(RMF3_data['Radius (km)'][10], RMF3_data['Solar Masses'][10]),
                xytext=(RMF3_data['Radius (km)'][10]+0.5, RMF3_data['Solar Masses'][10]+0.1),
                arrowprops=dict(arrowstyle="-"), color= cmap(norm(3)))

ax_mvr.annotate('SLY4', xy=(SLY4_data['Radius (km)'][10], SLY4_data['Solar Masses'][10]),
                xytext=(SLY4_data['Radius (km)'][10]+0.5, SLY4_data['Solar Masses'][10]+0.1),
                arrowprops=dict(arrowstyle="-"), color= cmap(norm(4)))

ax_mvr.annotate('GPPVA', xy=(GPPVA_data['Radius (km)'][20], GPPVA_data['Solar Masses'][20]),
                xytext=(GPPVA_data['Radius (km)'][20]+0.5, GPPVA_data['Solar Masses'][20]+0.1),
                arrowprops=dict(arrowstyle="-"), color= cmap(norm(5)))

ax_mvr.annotate('QHC19-A', xy=(QHC19A_data['Radius (km)'][10], QHC19A_data['Solar Masses'][10]),
                xytext=(QHC19A_data['Radius (km)'][10]+0.5, QHC19A_data['Solar Masses'][10]+0.1),
                arrowprops=dict(arrowstyle="-"), color= cmap(norm(6)))


plt.show()

fig, ax_lambda = plt.subplots(constrained_layout=True)
ax_lambda.set_xlabel(r'$\beta$')
ax_lambda.set_ylabel(r'$log_{10}\lambda (km^{5})$')
ax_lambda.set_ylim(4, 5)

ax_lambda.plot(RMF1_data['compactness'], polarizability_RMF1, color= cmap(norm(1)))
ax_lambda.plot(RMF2_data['compactness'], polarizability_RMF2, color= cmap(norm(2)))
ax_lambda.plot(RMF3_data['compactness'], polarizability_RMF3, color= cmap(norm(3)))
ax_lambda.plot(SLY4_data['compactness'], polarizability_SLY4, color= cmap(norm(4)))
ax_lambda.plot(GPPVA_data['compactness'], polarizability_GPPVA, color= cmap(norm(5)))
ax_lambda.plot(QHC19A_data['compactness'], polarizability_QHC19A, color= cmap(norm(6)))


ax_lambda.scatter(RMF1_data['compactness'][62], polarizability_RMF1[62], marker = "o", s = 10, color= 'black')
ax_lambda.scatter(RMF1_data['compactness'][28], polarizability_RMF1[28], facecolors='none', edgecolors='b', s = 15)
ax_lambda.annotate('RMF1', xy=(RMF1_data['compactness'][62], polarizability_RMF1[62]),
               xytext=(RMF1_data['compactness'][62]+0.01, polarizability_RMF1[62]+0.01),
               color= cmap(norm(1)))

ax_lambda.scatter(RMF2_data['compactness'][46], polarizability_RMF2[46], marker = "o", s = 10, color= 'black')
ax_lambda.scatter(RMF2_data['compactness'][22], polarizability_RMF2[22], facecolors='none', edgecolors='b', s = 15)
ax_lambda.annotate('RMF2', xy=(RMF2_data['compactness'][46], polarizability_RMF2[46]),
               xytext=(RMF2_data['compactness'][46]+0.01, polarizability_RMF2[46]+0.01),
               color= cmap(norm(2)))

ax_lambda.scatter(RMF3_data['compactness'][35], polarizability_RMF3[35], marker = "o", s = 10, color= 'black')
ax_lambda.scatter(RMF3_data['compactness'][17], polarizability_RMF3[17], facecolors='none', edgecolors='b', s = 15)
ax_lambda.annotate('RMF3', xy=(RMF3_data['compactness'][35], polarizability_RMF3[35]),
               xytext=(RMF3_data['compactness'][35]+0.01, polarizability_RMF3[35]+0.01),
               color= cmap(norm(3)))

ax_lambda.scatter(SLY4_data['compactness'][18], polarizability_SLY4[18], marker = "o", s = 10, color= 'black')
ax_lambda.scatter(SLY4_data['compactness'][8], polarizability_SLY4[8], facecolors='none', edgecolors='b', s = 15)
ax_lambda.annotate('SLY4', xy=(SLY4_data['compactness'][18], polarizability_SLY4[18]),
               xytext=(SLY4_data['compactness'][18]+0.01, polarizability_SLY4[18]+0.01),
               color= cmap(norm(4)))

ax_lambda.scatter(GPPVA_data['compactness'][13], polarizability_GPPVA[13], marker = "o", s = 10, color= 'black')
ax_lambda.scatter(GPPVA_data['compactness'][7], polarizability_GPPVA[7], facecolors='none', edgecolors='b', s = 15)
ax_lambda.annotate('GPPVA', xy=(QHC19A_data['compactness'][13], polarizability_GPPVA[13]),
               xytext=(GPPVA_data['compactness'][13]+0.01, polarizability_GPPVA[13]+0.01),
               color= cmap(norm(5)))

ax_lambda.scatter(QHC19A_data['compactness'][23], polarizability_QHC19A[23], marker = "o", s = 10, color= 'black')
ax_lambda.scatter(QHC19A_data['compactness'][11], polarizability_QHC19A[11], facecolors='none', edgecolors='b', s = 15)
ax_lambda.annotate('QHC19A', xy=(QHC19A_data['compactness'][23], polarizability_QHC19A[23]),
               xytext=(QHC19A_data['compactness'][23]+0.01, polarizability_QHC19A[23]+0.01),
               color= cmap(norm(6)))

plt.show()

fig, ax_lambda_r = plt.subplots(constrained_layout=True)
ax_lambda_r.set_xlabel(r'$R (km)$')
ax_lambda_r.set_ylabel(r'$log_{10}\lambda (km^{5})$')
ax_lambda_r.set_xlim(10, 15)
ax_lambda_r.set_ylim(3.3, 5)

ax_lambda_r.plot(RMF1_data['Radius (km)'], polarizability_RMF1, color= cmap(norm(1)))
ax_lambda_r.plot(RMF2_data['Radius (km)'], polarizability_RMF2, color= cmap(norm(2)))
ax_lambda_r.plot(RMF3_data['Radius (km)'], polarizability_RMF3, color= cmap(norm(3)))
ax_lambda_r.plot(SLY4_data['Radius (km)'], polarizability_SLY4, color= cmap(norm(4)))
ax_lambda_r.plot(GPPVA_data['Radius (km)'], polarizability_GPPVA, color= cmap(norm(5)))
ax_lambda_r.plot(QHC19A_data['Radius (km)'], polarizability_QHC19A, color= cmap(norm(6)))


ax_lambda_r.scatter(RMF1_data['Radius (km)'][62], polarizability_RMF1[62], marker = "o", s = 10, color= 'black')
ax_lambda_r.scatter(RMF1_data['Radius (km)'][28], polarizability_RMF1[28], facecolors='none', edgecolors='b', s = 15)
ax_lambda_r.annotate('RMF1', xy=(RMF1_data['Radius (km)'][100], polarizability_RMF1[100]),
               xytext=(RMF1_data['Radius (km)'][100]+0.01, polarizability_RMF1[100]+0.01),
               color= cmap(norm(1)))

ax_lambda_r.scatter(RMF2_data['Radius (km)'][46], polarizability_RMF2[46], marker = "o", s = 10, color= 'black')
ax_lambda_r.scatter(RMF2_data['Radius (km)'][22], polarizability_RMF2[22], facecolors='none', edgecolors='b', s = 15)
ax_lambda_r.annotate('RMF2', xy=(RMF2_data['Radius (km)'][100], polarizability_RMF2[100]),
               xytext=(RMF2_data['Radius (km)'][100]+0.01, polarizability_RMF2[100]+0.01),
               color= cmap(norm(2)))

ax_lambda_r.scatter(RMF3_data['Radius (km)'][35], polarizability_RMF3[35], marker = "o", s = 10, color= 'black')
ax_lambda_r.scatter(RMF3_data['Radius (km)'][17], polarizability_RMF3[17], facecolors='none', edgecolors='b', s = 15)
ax_lambda_r.annotate('RMF3', xy=(RMF3_data['Radius (km)'][100], polarizability_RMF3[100]),
               xytext=(RMF3_data['Radius (km)'][100]+0.01, polarizability_RMF3[100]+0.01),
               color= cmap(norm(3)))

ax_lambda_r.scatter(SLY4_data['Radius (km)'][18], polarizability_SLY4[18], marker = "o", s = 10, color= 'black')
ax_lambda_r.scatter(SLY4_data['Radius (km)'][8], polarizability_SLY4[8], facecolors='none', edgecolors='b', s = 15)
ax_lambda_r.annotate('SLY4', xy=(SLY4_data['Radius (km)'][100], polarizability_SLY4[100]),
               xytext=(SLY4_data['Radius (km)'][100]+0.01, polarizability_SLY4[100]+0.01),
               color= cmap(norm(4)))

ax_lambda_r.scatter(GPPVA_data['Radius (km)'][13], polarizability_GPPVA[13], marker = "o", s = 10, color= 'black')
ax_lambda_r.scatter(GPPVA_data['Radius (km)'][7], polarizability_GPPVA[7], facecolors='none', edgecolors='b', s = 15)
ax_lambda_r.annotate('GPPVA', xy=(QHC19A_data['Radius (km)'][100], polarizability_GPPVA[100]),
               xytext=(GPPVA_data['Radius (km)'][100]+0.01, polarizability_GPPVA[100]+0.01),
               color= cmap(norm(5)))

ax_lambda_r.scatter(QHC19A_data['Radius (km)'][23], polarizability_QHC19A[23], marker = "o", s = 10, color= 'black')
ax_lambda_r.scatter(QHC19A_data['Radius (km)'][11], polarizability_QHC19A[11], facecolors='none', edgecolors='b', s = 15)
ax_lambda_r.annotate('QHC19A', xy=(QHC19A_data['Radius (km)'][100], polarizability_QHC19A[100]),
               xytext=(QHC19A_data['Radius (km)'][100]+0.01, polarizability_QHC19A[100]+0.01),
               color= cmap(norm(6)))

plt.show()

import glob
import os
import numpy as np
from scipy.interpolate import interp1d

Kfix = 0.15
MpbhVal = 100
fpbh = 1e-3
zetaUV = 50
zetaX = 2e56
Tmin = 5e4
Nalpha = 4e3

dirName = 'PS_Files/PS_Files_Mpbh_{:.1e}_fpbh_{:.1e}_Xi_{:.0f}_Tmin_{:.3e}_Rfmp_15_chiX_{:.2e}_Nalpha_{:.2e}/'.format(MpbhVal,fpbh,zetaUV,Tmin,zetaX,Nalpha)
files = glob.glob(dirName + 'Red*')
listVals = []
for ff in files:
    z_strt = 10 + ff.find('/RedShift_')
    z_end = ff.find('_PS_mk')
    if z_end == -1:
            z_end = ff.find('_ps_no_')
    zval = float(ff[z_strt:z_end])
    vals = np.loadtxt(ff)[:,[0,1]]
    pspec = 10.**interp1d(np.log10(vals[:,0]), np.log10(vals[:,1]), kind='cubic')(np.log10(Kfix))
    listVals.append([zval, pspec])

np.savetxt('GenericPlotFiles/PowerSpec_Mpbh_{:.0f}_fpbh_{:.0e}_zetaUV_{:.0f}_zetaX_{:.1e}_Tmin_{:.1e}_Nalpha_{:.1e}_Kfix_{:.2f}.dat'.format(MpbhVal,fpbh,zetaUV,zetaX,Tmin,Nalpha,Kfix), listVals)


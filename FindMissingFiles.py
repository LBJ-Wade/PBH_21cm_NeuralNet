import os
import numpy
import glob

Mpbh=100

path = os.getcwd()
tbfiles = glob.glob('PS_Files/PS_Files_Mpbh_{:.1e}*'.format(Mpbh))

chiUV = [15, 30, 50, 90] 
chiX = [2e55, 2e56, 2e57]
Tmin = [1e4, 5e4, 1e5]
Nalph = [4e2, 4e3, 4e4]
fpbh = [1e-8,1e-6, 1e-4, 1e-2]

masterL = []
missingL = []

for chi1 in chiUV:
    for chi2 in chiX:
        for tmn in Tmin:
            for Na in Nalph:
                for fp in fpbh:
                    masterL.append([fp, chi1, chi2, tmn, Na])


for f in tbfiles:
    xeF = f.find('_Xi_')
    tminF = f.find('_Tmin_')
    chiF = f.find('_chiX_')
    NalF = f.find('_Nalpha_')
    fpbhF = f.find('_fpbh_')

    xi = float(f[xeF+4:tminF])
    tmin = float(f[tminF+6:tminF+15])
    chi = float(f[chiF+7:NalF])
    Nal = float(f[NalF+8:NalF+16])
    frac = float(f[fpbhF+6:xeF])

    if [frac, xi, chi, tmin, Nal] in masterL:
        masterL.remove([frac, xi, chi, tmin, Nal])
masterL.sort()
for vals in masterL:
    print vals


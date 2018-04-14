import numpy as np
import glob
import os

dirs = glob.glob('PS_Files/PS_Files*')

MpbhVal = 100
fullListFile = []
for dty in dirs:
    xi_st = dty.find('_Xi_')
    tmin_st = dty.find('_Tmin_')
    mpbh_st = dty.find('_Mpbh_')
    zetax_st = dty.find('_chiX_')
    nalph_st = dty.find('_Nalpha_')
    fpbh_st = dty.find('_fpbh_')

    nAlpha = float(dty[nalph_st + 8: nalph_st + 8 + 8])
    zetaX = float(dty[zetax_st + 6: nalph_st])
    tmin = float(dty[tmin_st + 6: tmin_st + 6 + 9])
    zetaUV = float(dty[xi_st + 4: tmin_st])
    mpbh = float(dty[mpbh_st + 6:fpbh_st])
    fpbh = float(dty[fpbh_st + 6: fpbh_st + 6 + 7])

    #print dty
    #print fpbh, mpbh, zetaUV, tmin, zetaX, nAlpha
    if MpbhVal != mpbh:
        continue

    xefactor = 1.07942599999999
    subfiles = glob.glob(dty + '/Tb_Xi_*.dat')
    if len(subfiles) > 0:
        for ff in subfiles:
            vals = np.loadtxt(ff)
            np.savetxt('XeFiles/xe_file_mpbh_{:.0e}_fpbh_{:.0e}_zetaUV_{:.0f}_zetaX_{:.0e}_Tmin_{:.1e}_Nalpha_{:.0e}.dat'.format(mpbh,fpbh,zetaUV, zetaX,tmin,nAlpha), np.column_stack((vals[:,0], (1.- vals[:,1])*xefactor)))
            for zz in range(len(vals[:,0])):
                fullListFile.append([vals[zz,0], np.log10(fpbh), np.log10(zetaUV), np.log10(zetaX), np.log10(tmin), np.log10(nAlpha),  (1.- vals[zz, 1])*xefactor])
    else:
        subfiles = glob.glob(dty + '/RedShift*')
        zzList = []
        xeList = []
        for ff in subfiles:
            zz = float(ff[ff.find('/RedShift_')+10:ff.find('_ps_no_halos')])

        
            xe = float(ff[ff.find('_nf')+3:ff.find('_useTs1_')])
            xeval = (1. - xe)*xefactor
            fullListFile.append([zz, np.log10(fpbh), np.log10(zetaUV), np.log10(zetaX), np.log10(tmin), np.log10(nAlpha),  xeval])
            zzList.append(zz)
            xeList.append(xeval)
        np.savetxt('XeFiles/xe_file_mpbh_{:.0e}_fpbh_{:.0e}_zetaUV_{:.0f}_zetaX_{:.0e}_Tmin_{:.1e}_Nalpha_{:.0e}.dat'.format(mpbh,fpbh,zetaUV, zetaX,tmin, nAlpha),
                   np.column_stack((zzList, xeList)))

fullListFileHold = set(map(tuple,fullListFile))
fullListFile = map(list,fullListFileHold)
np.savetxt('XeFiles/XeFull_Mpbh_{:.0e}.dat'.format(MpbhVal), fullListFile, fmt='%.3e')

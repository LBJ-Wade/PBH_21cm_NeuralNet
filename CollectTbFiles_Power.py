import numpy as np
import glob
import os
import math

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
    
    #subfiles = glob.glob(dty + '/*')
    subfiles = glob.glob(dty + '/RedShift_*.dat')
    for ff in subfiles:
        vals = np.loadtxt(ff)
        z_strt = 10 + ff.find('/RedShift_')
        z_end = ff.find('_PS_mk')
        zval = float(ff[z_strt:z_end])

        for j in range(len(vals[:,0])):
            if math.isnan(vals[j, 1]):
                output = 0.
            else:
                output = vals[j, 1]
            fullListFile.append([zval, vals[j, 0], np.log10(fpbh), np.log10(zetaX), np.log10(tmin), np.log10(nAlpha),  output])

np.savetxt('TbFiles/TbFull_Power_Mpbh_{:.0e}.dat'.format(MpbhVal), fullListFile, fmt='%.3e')

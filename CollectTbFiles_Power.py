import numpy as np
import glob
import os
import math

dirs = glob.glob('PS_Files/PS_Files*')

#Zfix = 17.57 # must be one of the values already computed
Z_list = [8.38, 8.85, 9.34, 9.86, 10.40, 10.97, 11.57, 12.20, 12.86, 13.55,
          14.28, 15.05, 15.85, 16.69, 17.57, 18.50, 19.48]

MpbhVal = 100
for Zfix in Z_list:
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
        sve_vec = []
        subfiles = glob.glob(dty + '/RedShift_*')

        for ff in subfiles:
            vals = np.loadtxt(ff)
            z_strt = 10 + ff.find('/RedShift_')
            
            z_end = ff.find('_PS_mk')
            if z_end == -1:
                z_end = ff.find('_ps_no_')
            zval = float(ff[z_strt:z_end])
            if len(vals) == 0:
                print dty, ff
                exit()
            if zval != Zfix:
                appendFull = False
                if os.path.isfile('TbFiles/tb_PowerSpectrum_mpbh_{:.0e}_fpbh_{:.0e}_zetaUV_{:.0f}_zetaX_{:.0e}_Tmin_{:.1e}_Nalpha_{:.0e}.dat'.format(mpbh,fpbh,zetaUV,
                    zetaX,tmin,nAlpha)):
                    #continue
                    pass
            else:
                appendFull = True
            for j in range(len(vals[:,0])):
                if math.isnan(vals[j, 1]) or vals[j,1] < 0.:
                    continue
                if appendFull:
                    fullListFile.append([np.log10(vals[j, 0]), np.log10(fpbh), np.log10(zetaUV), np.log10(zetaX), np.log10(tmin), np.log10(nAlpha),  vals[j,1]])
                sve_vec.append([zval, vals[j, 0], vals[j,1]])
     
        np.savetxt('TbFiles/tb_PowerSpectrum_mpbh_{:.0e}_fpbh_{:.0e}_zetaUV_{:.0f}_zetaX_{:.0e}_Tmin_{:.1e}_Nalpha_{:.0e}.dat'.format(mpbh,fpbh,zetaUV,
                    zetaX,tmin,nAlpha), sve_vec)

    np.savetxt('TbFiles/TbFull_Power_Mpbh_{:.0e}_Zval_{:.2f}.dat'.format(MpbhVal, Zfix), fullListFile, fmt='%.3e')

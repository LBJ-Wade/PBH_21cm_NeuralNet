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
    
    #subfiles = glob.glob(dty + '/*')
    subfiles = glob.glob(dty + '/Tb_Xi_*.dat')
    if len(subfiles) > 0:
        for ff in subfiles:
            vals = np.loadtxt(ff)
            np.savetxt('TbFiles/tb_file_mpbh_{:.0e}_fpbh_{:.0e}_zetaUV_{:.0f}_zetaX_{:.0e}_Tmin_{:.1e}_Nalpha_{:.0e}.dat'.format(mpbh,fpbh,zetaUV,
                                                                                                                                 zetaX,tmin,nAlpha), vals[:,[0,2]])
            for zz in range(len(vals[:,0])):
                fullListFile.append([vals[zz,0], np.log10(fpbh), np.log10(zetaUV), np.log10(zetaX), np.log10(tmin), np.log10(nAlpha),  vals[zz, 2]])
    else:
        subfiles = glob.glob(dty + '/RedShift*')
        zzList = []
        tbList = []
        for ff in subfiles:
            zendV = ff.find('_ps_no_halos')
            if zendV == -1:
                zendV = ff.find('_PS_mk_')
            try:
                zz = float(ff[ff.find('/RedShift_')+10:zendV])
                tb = float(ff[ff.find('aveTb')+5:ff.find('_Pop2')])
            except ValueError:
                continue
                
            fullListFile.append([zz, np.log10(fpbh), np.log10(zetaUV), np.log10(zetaX), np.log10(tmin), np.log10(nAlpha),  tb])
            zzList.append(zz)
            tbList.append(tb)
        np.savetxt('TbFiles/tb_file_mpbh_{:.0e}_fpbh_{:.0e}_zetaUV_{:.0f}_zetaX_{:.0e}_Tmin_{:.1e}_Nalpha_{:.0e}.dat'.format(mpbh,fpbh,zetaUV, zetaX,tmin,nAlpha),
                   np.column_stack((zzList, tbList)))

fullListFileHold = set(map(tuple,fullListFile))
fullListFile = map(list,fullListFileHold)
np.savetxt('TbFiles/TbFull_Mpbh_{:.0e}.dat'.format(MpbhVal), fullListFile, fmt='%.3e')

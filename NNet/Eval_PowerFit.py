import numpy as np
import os
from scipy.interpolate import interp2d
from Tb_PBH_ANN import *

arrayName = 'hera127'
arrayErr = np.loadtxt('../Sensitivities/NoiseVals_'+arrayName+'.dat')
sensty_arr = interp2d(arrayErr[:,0], arrayErr[:,1], arrayErr[:,2], kind='linear', bounds_error=False, fill_value=1e4)

tb_analysis = True
GlobalTb = True

Mpbh = 100
Nhidden = 50

Pts_perVar = 3
fpbh_L = np.logspace(-7, -2, Pts_perVar)
zetaUV_L = np.linspace(15, 90, Pts_perVar)
zetaX_L = np.logspace(np.log10(2e55), np.log10(2e57), Pts_perVar)
Tmin_L = np.logspace(4, 5, Pts_perVar)
Nalpha_L = np.logspace(np.log10(4e2), np.log10(4e4), Pts_perVar)
k_List = np.logspace(0.1, 2, Pts_perVar)
#Z_list = [8.38, 8.85, 9.34, 9.86, 10.40, 10.97, 11.57, 12.20, 12.86, 13.55,
#          14.28, 15.05, 15.85, 16.69, 17.57, 18.50, 19.48]
Z_list = [17.57]


chi2_list = []
param_list = []

for fp in fpbh_L:
    for zUV in zetaUV_L:
        for zX in zetaX_L:
            for Tm in Tmin_L:
                for Na in Nalpha_L:
                    chi2 = 0.
                    param_list.append([fp, zUV, zX, Tm, Na])
                    for zz in Z_list:
                        initPBH = Tb_PBH_Nnet(Mpbh, globalTb=GlobalTb, HiddenNodes=Nhidden, zfix=zz)
                        initPBH.main_nnet()
                        true_list = []
                        eval_list = []
                        error = []
                        for i,kk in enumerate(k_List):
                            true_list.append([np.log10(kk), -8, np.log10(50), np.log10(2e56), np.log10(5e4), np.log10(4e3)])
                            eval_list.append([np.log10(kk), np.log10(fp), np.log10(zUV), np.log10(zX), np.log10(Tm), np.log10(Na)])
                            error.append(sensty_arr(zz, kk))
                        val = initPBH.solo_eval(eval_list)
                        trueV = initPBH.solo_eval(true_list)
                        chi2 += np.sum(((val - trueV) / error)**2.)
                    print param_list[-1], chi2
                    chi2_list.append(chi2)

chi2_list = np.asarray(chi2_list)
param_list = np.asarray(param_list)
np.savetxt('../Sensitivities/Chi2_Fits_' + arrayName + '_TbPower_Mpbh_{:.0f}_ModerateSense.dat', np.column_stack((param_list, chi2_list)))




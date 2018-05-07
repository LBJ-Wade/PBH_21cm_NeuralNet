import numpy as np
import os
import tensorflow as tf
from ImportTbPower import *
import itertools
from Tb_PBH_ANN import *
from scipy.optimize import minimize
from scipy.interpolate import interp2d

arrayName = 'SKA'
arrayErr = np.loadtxt('../Sensitivities/NoiseVals_'+arrayName+'.dat')
sensty_arr = interp2d(arrayErr[:,0], arrayErr[:,1], arrayErr[:,2], kind='linear', bounds_error=False, fill_value=1e5)
hlittle = 0.7
tb_analysis = True
GlobalTb = False

Mpbh = 100
Nhidden = 50

#fpbhL = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
fpbhL = np.logspace(-8, -2, 20)

k_List = np.logspace(np.log10(0.15), np.log10(1), 10.)
Z_list = [8.38, 8.85, 9.34, 9.86, 10.40, 10.97, 11.57, 12.20, 12.86, 13.55,
          14.28, 15.05, 15.85, 16.69, 17.57, 18.50, 19.48]

bndVals = []
sve_paramsBF = []
chiSqV = 3.95


modeler = np.zeros(len(Z_list), dtype=object)
error = np.zeros((len(Z_list), len(k_List)))
true_list = np.zeros((len(Z_list), len(k_List)))
for j,zz in enumerate(Z_list):
    initPBH = Tb_PBH_Nnet(Mpbh, globalTb=GlobalTb, HiddenNodes=Nhidden, zfix=zz)
    initPBH.main_nnet()
    initPBH.load_matrix_elems()
    vechold = []
    for i,kk in enumerate(k_List):
        error[j,i] = sensty_arr(zz, kk/hlittle)
        vechold.append([np.log10(kk), -8., np.log10(50), np.log10(2e56), np.log10(5e4), np.log10(4e3)])
    true_list[j,:] = list(itertools.chain.from_iterable(initPBH.rapid_eval(vechold)))
    error[j,:] = np.sqrt(error[j,:]**2. + (0.4*true_list[j,:])**2.)

    modeler[j] = ImportGraph(initPBH.fileN, Mpbh, zz)

def min_wrapper(x, fpbh=1e-8):
    zUV, zX, Tm, Na = x
    chi2 = 0.
    for j,zz in enumerate(Z_list):
        eval_list = []
        for i,kk in enumerate(k_List):
            eval_list.append([np.log10(kk), np.log10(fpbh), np.log10(zUV), np.log10(zX), np.log10(Tm), np.log10(Na)])
        val = modeler[j].run_yhat(eval_list)
        chi2 += np.sum(((val.flatten() - true_list[j,:]) / error[j,:])**2.)
    return chi2

guessX0 = [50, 2e56, 5e4, 4e3]

for fpbh in fpbhL:
    print 'Testing fpbh {:.0e}'.format(fpbh)
    minSLN = minimize(lambda x: min_wrapper(x, fpbh=fpbh), guessX0)
    #print minSLN
    bndVals.append(minSLN.fun)
    sve_paramsBF.append(minSLN.x)
    
    
np.savetxt('../Sensitivities/Frequentist_Mpbh_{:.0e}_'.format(Mpbh) + arrayName + '.dat', np.column_stack((fpbhL, np.asarray(bndVals))))

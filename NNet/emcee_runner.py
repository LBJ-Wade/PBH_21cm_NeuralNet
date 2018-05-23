import numpy as np
import tensorflow as tf
import emcee
from scipy.interpolate import interp2d
import os
from Tb_PBH_ANN import *
import itertools
import matplotlib
matplotlib.use('Agg')
import pylab as pl
import matplotlib.pyplot as plt
import corner
from matplotlib import rc
from ImportTbPower import *
rc('font',**{'family':'serif','serif':['Times','Palatino']})
rc('text', usetex=True)


hlittle = 0.7
arrayName = 'SKA'
arrayErr = np.loadtxt('../Sensitivities/NoiseVals_'+arrayName+'.dat')
sensty_arr = interp2d(arrayErr[:,0], arrayErr[:,1], arrayErr[:,2], kind='linear', bounds_error=False, fill_value=1e5)

Mpbh = 100
ln_fpbhMAX = -2
Nhidden = 50
BurnPTS = 500
NSTEPS = 1e6
ndim, nwalkers = 5, 100

filePTS = 'mcmc_pts/MCMC_pts_Mpbh_{:.0f}_'.format(Mpbh)+arrayName+'_.dat'
scterPlt = 'mcmc_pts/MCMC_PLT_Mpbh_{:.0f}_'.format(Mpbh)+arrayName+'_.pdf'
cornerPLT = 'mcmc_pts/Corner_Mpbh_{:.0f}_'.format(Mpbh)+arrayName+'_.pdf'

k_List = np.logspace(np.log10(0.1), np.log10(2), 15.)
#Z_list = [8.38, 8.85, 9.34, 9.86, 10.40, 10.97, 11.57, 12.20, 12.86, 13.55,
#          14.28, 15.05, 15.85, 16.69, 17.57, 18.50, 19.48]
Z_list = [12.86]

init_params = [-7., 1.6, np.log10(2e56), np.log10(5e4), np.log10(4e3)]
params_low = [-8., np.log10(15), np.log10(2e55), np.log10(1e4), np.log10(4e2)]
params_space = [ln_fpbhMAX - params_low[0], np.log10(90) - np.log10(15), 2, 1, 2]
Truth_params =  [-7.95, np.log10(50), np.log10(2e56), np.log10(5e4), np.log10(4e3)] #actually fpbh = -8

modeler = np.zeros(len(Z_list), dtype=object)
error = np.zeros((len(Z_list), len(k_List)))
true_list = np.zeros((len(Z_list), len(k_List)))
for j,zz in enumerate(Z_list):
    initPBH = Tb_PBH_Nnet(Mpbh, globalTb=False, HiddenNodes=Nhidden, zfix=zz)
    initPBH.main_nnet()
    initPBH.load_matrix_elems()
    vechold = []
    for i,kk in enumerate(k_List):
        error[j,i] = sensty_arr(zz, kk/hlittle)
        vechold.append([np.log10(kk), -8., np.log10(50), np.log10(2e56), np.log10(5e4), np.log10(4e3)])
    true_list[j,:] = list(itertools.chain.from_iterable(initPBH.rapid_eval(vechold)))
    #error[j,:] = np.sqrt(error[j,:]**2. + (0.3*true_list[j,:])**2.)

    modeler[j] = ImportGraph(initPBH.fileN, Mpbh, zz)

def lnprior(theta):
    lnf, lnUV, lnX, lnT, lnN = theta
    if (-8 < lnf < ln_fpbhMAX and np.log10(15) < lnUV < np.log10(90) and np.log10(2e55) < lnX < np.log10(2e57)
        and 4 < lnT < 5 and np.log10(4e2) < lnN < np.log10(4e4)):
        return 0.
    else:
        return -np.inf

def ln_like(theta):
    lnf, lnUV, lnX, lnT, lnN = theta
    evalPts = np.zeros((len(Z_list), len(k_List)))
    for j,zz in enumerate(Z_list):
        eval_list = []
        for i,kk in enumerate(k_List):
            eval_list.append([np.log10(kk), lnf, lnUV, lnX, lnT, lnN])
        evalPts[j,:] = list(itertools.chain.from_iterable(modeler[j].run_yhat(eval_list)))
    return -np.sum(((evalPts-true_list)/error)**2.)

def lnprob(theta):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_like(theta)

#pos = [init_params + 1e-1*np.random.randn(ndim) for i in range(nwalkers)]
pos = np.asarray([params_low + params_space*np.random.rand(ndim) for i in range(nwalkers)])
pos[:,0] = -7

print 'Running Sampler.'
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=1)
#sampler.run_mcmc(pos, NSTEPS)

#f = open("chain_mpbh_{:.0e}.dat".format(Mpbh), "w")
#f.close()

nsteps = 1e5
for i, result in enumerate(sampler.sample(pos, iterations=nsteps)):
    if (i+1) % 100 == 0:
        print("{0:5.1%}".format(float(i) / nsteps))

#try:
#    print 'Autocorrelation Time: ', sampler.get_autocorr_time()
#except:
#    pass

print 'Making Plots...'
#print 'Making burn-in plot...'
#fig, axes = plt.subplots(5, 1, sharex=True)
#for j in range(ndim):
#    for i in range(nwalkers):
#        axes[j].plot(sampler.chain[i,:,j], lw=1, color='k',alpha=0.3)
#        if j == 0:
#            axes[j].set_ylim([-8, ln_fpbhMAX])
#        if j == 1:
#            axes[j].set_ylim([1, 2])
#        if j == 2:
#            axes[j].set_ylim([55, 57])
#        if j == 3:
#            axes[j].set_ylim([4, 5])
#        if j == 4:
#            axes[j].set_ylim([2, 5])
#plt.savefig(scterPlt)

samples = sampler.chain[:, BurnPTS:, :].reshape((-1, ndim))

print '2sigma limit: ', np.percentile(samples, [95])

fig = corner.corner(samples, labels=[r"$f_{pbh}$", r"$\zeta_{UV}$", r"$\zeta_X$", "$T$", r"$N_{\alpha}$"],
                      truths=Truth_params, color='k', quantiles=[0.16, 0.84, 0.95],
                      show_titles=True, title_kwargs={"fontsize": 12})
fig.savefig(cornerPLT)

np.savetxt(filePTS, samples)


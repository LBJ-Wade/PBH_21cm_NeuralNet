import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab as pl
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import matplotlib.patheffects as PathEffects
import matplotlib.gridspec as gridspec
import glob
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib import rc
from Tb_PBH_ANN import *
from Xe_PBH_ANN import *

#rc('font',**{'family':'serif','serif':['Palatino','Palatino']})
#rc('text', usetex=True)
#rc('font',**{'family':'Apple Chancery'})

mpl.rcParams['xtick.major.size']=8
mpl.rcParams['ytick.major.size']=8
mpl.rcParams['xtick.labelsize']=12
mpl.rcParams['ytick.labelsize']=12
fs=16

path = os.getcwd()

# Set global parameters and init NN
# Define type of anlaysis to be used and PBH mass
test_plots = os.getcwd() + '/Test_Plots/'

# DEFINE ANALYSIS
tb_analysis = True
GlobalTb = False
######

Mpbh = 100
Nhidden = 50

Zlist = np.linspace(6, 35, 70)
klist = np.logspace(np.log10(0.05), np.log10(2), 70)
color_list = ['#9883E5', '#72A1E5', '#50C9CE', '#2E382E',
              '#7B1E7A', '#F9564F', '#F3C677', '#44AF69']
Zpower = 8.38

fpbh = 1e-4
zetaUV = 90
zetaX = 2e55
Tmin = 5e4
Nalpha = 4e3


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Begin Plot Making
if tb_analysis:
    initPBH = Tb_PBH_Nnet(Mpbh, globalTb=GlobalTb, HiddenNodes=Nhidden, zfix=Zpower)
    initPBH.main_nnet()
    ftag = 'Tb'
    if not GlobalTb:
        ftag += '_Power_'
else:
    initPBH = Xe_PBH_Nnet(Mpbh,HiddenNodes=Nhidden)
    initPBH.main_nnet()
    ftag = 'Xe'


def power_spectrum_pull(file, zVal):
    data = np.loadtxt(file)
    #print zVal
    #print data
    indxs = np.isclose(data[:,0], zVal, atol=1e-2)
    data = data[indxs]
    #print data
    #exit()
    return np.column_stack((data[:,1], data[:,2]))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Scan over fpbh
print 'Creating fpbh plot...'
pl.figure()
ax = pl.gca()
filename = test_plots + '/Test_' + ftag + '_fpbh_scan_zetaUV_{:.0f}_zetaX_{:.0e}_Tmin_{:.1e}_Nalpha_{:.0e}.pdf'.format(zetaUV, zetaX, Tmin, Nalpha)

fpbhL = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

evalVec = []
for f in fpbhL:
    if not tb_analysis or GlobalTb:
        for z in Zlist:
            evalVec.append([z, np.log10(f), np.log10(zetaUV),
                            np.log10(zetaX), np.log10(Tmin), np.log10(Nalpha)])
    else:
        for kk in klist:
            evalVec.append([np.log10(kk), np.log10(f), np.log10(zetaUV),
                            np.log10(zetaX), np.log10(Tmin), np.log10(Nalpha)])
vals = initPBH.eval_NN(evalVec=evalVec)


for i,f in enumerate(fpbhL):
    inarray = np.asarray(evalVec)
    yvals = vals[inarray[:, -5] == np.log10(f)]
    xvals = inarray[inarray[:, -5] == np.log10(f)][:,0]
    if tb_analysis and not GlobalTb:
        xvals = np.power(10, xvals)

    yvals = yvals[np.argsort(xvals)]
    xvals.sort()
    
    pl.plot(xvals, yvals, lw=1, color=color_list[i], label=r'$f_{pbh} = $'+'{:.0e}'.format(f))
    if tb_analysis:
        if GlobalTb:
            tbfile = '../TbFiles/tb_file_mpbh_{:.0e}_fpbh_{:.0e}_zetaUV_{:.0f}_zetaX_{:.0e}_Tmin_{:.1e}_Nalpha_{:.0e}.dat'.format(Mpbh,f,zetaUV,zetaX,Tmin,Nalpha)
        else:
            tbfile = '../TbFiles/tb_PowerSpectrum_mpbh_{:.0e}_fpbh_{:.0e}_zetaUV_{:.0f}_zetaX_{:.0e}_Tmin_{:.1e}_Nalpha_{:.0e}.dat'.format(Mpbh,f,zetaUV,zetaX,Tmin,Nalpha)
    else:
        tbfile = '../XeFiles/xe_file_mpbh_{:.0e}_fpbh_{:.0e}_zetaUV_{:.0f}_zetaX_{:.0e}_Tmin_{:.1e}_Nalpha_{:.0e}.dat'.format(Mpbh,f,zetaUV,zetaX,Tmin,Nalpha)
    if os.path.exists(tbfile):
        if not tb_analysis or GlobalTb:
            sim_data = np.loadtxt(tbfile)
        else:
            sim_data = power_spectrum_pull(tbfile, Zpower)

        pl.plot(sim_data[:,0], sim_data[:,1], 'o', ms=3, lw=1, color=color_list[i])
ax.axhline(y=0, xmin=0, xmax=50, color='k')

if tb_analysis:
    if GlobalTb:
        ax.set_ylabel(r'$\delta T_b$  [mK]', fontsize=fs)
        plt.xlim([6,30])
        ax.set_xlabel(r'z', fontsize=fs)
    else:
        ax.set_ylabel(r'$\Delta T_b$ ', fontsize=fs)
        ax.set_yscale('log')
        ax.set_xscale('log')
        plt.xlim([0.05, 2.])
        plt.ylim([0.1, 4e3])
        ax.set_xlabel(r'k', fontsize=fs)
else:
    ax.set_ylabel(r'$x_e$', fontsize=fs)
    plt.xlim([6,30])
    ax.set_xlabel(r'z', fontsize=fs)
plt.legend()
plt.tight_layout()
plt.savefig(filename)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Scan over Nalpha
print 'Creating Nalpha plot...'
pl.figure()
ax = pl.gca()
filename = test_plots + '/Test_' + ftag + '_Nalpha_scan_zetaUV_{:.0f}_zetaX_{:.0e}_Tmin_{:.1e}_fpbh_{:.0e}.pdf'.format(zetaUV, zetaX, Tmin, fpbh)

NalphaL = [4e2, 1e3, 4e3, 1e4, 4e4]

evalVec = []
for na in NalphaL:
    if not tb_analysis or GlobalTb:
        for z in Zlist:
            evalVec.append([z, np.log10(fpbh), np.log10(zetaUV),
                            np.log10(zetaX), np.log10(Tmin), np.log10(na)])

    else:
        for kk in klist:
            evalVec.append([np.log10(kk), np.log10(fpbh), np.log10(zetaUV),
                            np.log10(zetaX), np.log10(Tmin), np.log10(na)])
vals = initPBH.eval_NN(evalVec=evalVec)

for i,na in enumerate(NalphaL):
    inarray = np.asarray(evalVec)
    
    yvals = vals[inarray[:, -1] == np.log10(na)]
    xvals = inarray[inarray[:, -1] == np.log10(na)][:,0]
    if tb_analysis and not GlobalTb:
        xvals = np.power(10, xvals)
    yvals = yvals[np.argsort(xvals)]
    xvals.sort()
    pl.plot(xvals, yvals, lw=1, color=color_list[i], label=r'$N_\alpha = $'+'{:.1e}'.format(na))
    if tb_analysis:
        if GlobalTb:
            tbfile = '../TbFiles/tb_file_mpbh_{:.0e}_fpbh_{:.0e}_zetaUV_{:.0f}_zetaX_{:.0e}_Tmin_{:.1e}_Nalpha_{:.0e}.dat'.format(Mpbh,fpbh,zetaUV,zetaX,Tmin,na)
        else:
            tbfile = '../TbFiles/tb_PowerSpectrum_mpbh_{:.0e}_fpbh_{:.0e}_zetaUV_{:.0f}_zetaX_{:.0e}_Tmin_{:.1e}_Nalpha_{:.0e}.dat'.format(Mpbh,fpbh,zetaUV,zetaX,Tmin,na)
    else:
        tbfile = '../XeFiles/xe_file_mpbh_{:.0e}_fpbh_{:.0e}_zetaUV_{:.0f}_zetaX_{:.0e}_Tmin_{:.1e}_Nalpha_{:.0e}.dat'.format(Mpbh,fpbh,zetaUV,zetaX,Tmin,na)
    if os.path.exists(tbfile):
        if not tb_analysis or GlobalTb:
            sim_data = np.loadtxt(tbfile)
        else:
            sim_data = power_spectrum_pull(tbfile, Zpower)
        pl.plot(sim_data[:,0], sim_data[:,1], 'o', ms=3, lw=1, color=color_list[i])

ax.axhline(y=0, xmin=0, xmax=50, color='k')
if tb_analysis:
    if GlobalTb:
        ax.set_ylabel(r'$\delta T_b$  [mK]', fontsize=fs)
        plt.xlim([6,30])
        ax.set_xlabel(r'z', fontsize=fs)
    else:
        ax.set_ylabel(r'$\Delta T_b$ ', fontsize=fs)
        ax.set_yscale('log')
        ax.set_xscale('log')
        plt.xlim([0.05, 2.])
        plt.ylim([0.1, 4e3])
        ax.set_xlabel(r'k', fontsize=fs)
else:
    ax.set_ylabel(r'$x_e$', fontsize=fs)
    plt.xlim([6,30])
    ax.set_xlabel(r'z', fontsize=fs)
plt.legend()
plt.tight_layout()
plt.savefig(filename)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Scan over Tmin
print 'Creating Tmin plot...'
pl.figure()
ax = pl.gca()
filename = test_plots + '/Test_' + ftag + '_Tmin_scan_zetaUV_{:.0f}_zetaX_{:.0e}_Nalpha_{:.0e}_fpbh_{:.0e}.pdf'.format(zetaUV, zetaX, Nalpha, fpbh)

TminL = [1e4, 2.5e4, 5e4, 7e4, 1e5]

evalVec = []
for tm in TminL:
    if not tb_analysis or GlobalTb:
        for z in Zlist:
            evalVec.append([z, np.log10(fpbh), np.log10(zetaUV),
                            np.log10(zetaX), np.log10(tm), np.log10(Nalpha)])

    else:
        for kk in klist:
            evalVec.append([np.log10(kk), np.log10(fpbh), np.log10(zetaUV),
                            np.log10(zetaX), np.log10(tm), np.log10(Nalpha)])
vals = initPBH.eval_NN(evalVec=evalVec)


for i,tm in enumerate(TminL):
    inarray = np.asarray(evalVec)
    
    yvals = vals[inarray[:, -2] == np.log10(tm)]
    xvals = inarray[inarray[:, -2] == np.log10(tm)][:,0]
    if tb_analysis and not GlobalTb:
        xvals = np.power(10, xvals)
    yvals = yvals[np.argsort(xvals)]
    xvals.sort()
    pl.plot(xvals, yvals, lw=1, color=color_list[i], label=r'$T^{vir}_{min} = $'+'{:.1e}'.format(tm))
    if tb_analysis:
        if GlobalTb:
            tbfile = '../TbFiles/tb_file_mpbh_{:.0e}_fpbh_{:.0e}_zetaUV_{:.0f}_zetaX_{:.0e}_Tmin_{:.1e}_Nalpha_{:.0e}.dat'.format(Mpbh,fpbh,zetaUV,zetaX,tm,Nalpha)
        else:
            tbfile = '../TbFiles/tb_PowerSpectrum_mpbh_{:.0e}_fpbh_{:.0e}_zetaUV_{:.0f}_zetaX_{:.0e}_Tmin_{:.1e}_Nalpha_{:.0e}.dat'.format(Mpbh,fpbh,zetaUV,zetaX,tm,Nalpha)
    else:
        tbfile = '../XeFiles/xe_file_mpbh_{:.0e}_fpbh_{:.0e}_zetaUV_{:.0f}_zetaX_{:.0e}_Tmin_{:.1e}_Nalpha_{:.0e}.dat'.format(Mpbh,fpbh,zetaUV,zetaX,tm,Nalpha)
    if os.path.exists(tbfile):
        if not tb_analysis or GlobalTb:
            sim_data = np.loadtxt(tbfile)
        else:
            sim_data = power_spectrum_pull(tbfile, Zpower)
        pl.plot(sim_data[:,0], sim_data[:,1], 'o', ms=3, lw=1, color=color_list[i])

ax.axhline(y=0, xmin=0, xmax=50, color='k')
if tb_analysis:
    if GlobalTb:
        ax.set_ylabel(r'$\delta T_b$  [mK]', fontsize=fs)
        plt.xlim([6,30])
        ax.set_xlabel(r'z', fontsize=fs)
    else:
        ax.set_ylabel(r'$\Delta T_b$ ', fontsize=fs)
        ax.set_yscale('log')
        ax.set_xscale('log')
        plt.xlim([0.05, 2.])
        plt.ylim([0.1, 4e3])
        ax.set_xlabel(r'k', fontsize=fs)
else:
    ax.set_ylabel(r'$x_e$', fontsize=fs)
    plt.xlim([6,30])
    ax.set_xlabel(r'z', fontsize=fs)
plt.legend()
plt.tight_layout()
plt.savefig(filename)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Scan over zetaX
print 'Creating Zeta X plot...'
pl.figure()
ax = pl.gca()
filename = test_plots + '/Test_' + ftag + '_ZetaX_scan_zetaUV_{:.0f}_Tmin_{:.1e}_Nalpha_{:.0e}_fpbh_{:.0e}.pdf'.format(zetaUV, Tmin, Nalpha, fpbh)

zetaXL = [2e55, 7e55, 2e56, 7e56, 2e57]

evalVec = []
for zx in zetaXL:
    if not tb_analysis or GlobalTb:
        for z in Zlist:
            evalVec.append([z, np.log10(fpbh), np.log10(zetaUV),
                            np.log10(zx), np.log10(Tmin), np.log10(Nalpha)])

    else:
        for kk in klist:
            evalVec.append([np.log10(kk), np.log10(fpbh), np.log10(zetaUV),
                            np.log10(zx), np.log10(Tmin), np.log10(Nalpha)])
vals = initPBH.eval_NN(evalVec=evalVec)

for i,zx in enumerate(zetaXL):
    inarray = np.asarray(evalVec)
    
    yvals = vals[inarray[:, -3] == np.log10(zx)]
    xvals = inarray[inarray[:, -3] == np.log10(zx)][:,0]
    if tb_analysis and not GlobalTb:
        xvals = np.power(10, xvals)
    yvals = yvals[np.argsort(xvals)]
    xvals.sort()
    pl.plot(xvals, yvals, lw=1, color=color_list[i], label=r'$\zeta_{X} = $'+'{:.1e}'.format(zx))
    if tb_analysis:
        if GlobalTb:
            tbfile = '../TbFiles/tb_file_mpbh_{:.0e}_fpbh_{:.0e}_zetaUV_{:.0f}_zetaX_{:.0e}_Tmin_{:.1e}_Nalpha_{:.0e}.dat'.format(Mpbh,fpbh,zetaUV,zx,Tmin,Nalpha)
        else:
            tbfile = '../TbFiles/tb_PowerSpectrum_mpbh_{:.0e}_fpbh_{:.0e}_zetaUV_{:.0f}_zetaX_{:.0e}_Tmin_{:.1e}_Nalpha_{:.0e}.dat'.format(Mpbh,fpbh,zetaUV,zx,Tmin,Nalpha)
    else:
        tbfile = '../XeFiles/xe_file_mpbh_{:.0e}_fpbh_{:.0e}_zetaUV_{:.0f}_zetaX_{:.0e}_Tmin_{:.1e}_Nalpha_{:.0e}.dat'.format(Mpbh,fpbh,zetaUV,zx,Tmin,Nalpha)
    if os.path.exists(tbfile):
        if not tb_analysis or GlobalTb:
            sim_data = np.loadtxt(tbfile)
        else:
            sim_data = power_spectrum_pull(tbfile, Zpower)
        pl.plot(sim_data[:,0], sim_data[:,1], 'o', ms=3, lw=1, color=color_list[i])

ax.axhline(y=0, xmin=0, xmax=50, color='k')
if tb_analysis:
    if GlobalTb:
        ax.set_ylabel(r'$\delta T_b$  [mK]', fontsize=fs)
        plt.xlim([6,30])
        ax.set_xlabel(r'z', fontsize=fs)
    else:
        ax.set_ylabel(r'$\Delta T_b$ ', fontsize=fs)
        ax.set_yscale('log')
        ax.set_xscale('log')
        plt.xlim([0.05, 2.])
        plt.ylim([0.1, 4e3])
        ax.set_xlabel(r'k', fontsize=fs)
else:
    ax.set_ylabel(r'$x_e$', fontsize=fs)
    plt.xlim([6,30])
    ax.set_xlabel(r'z', fontsize=fs)
plt.legend()
plt.tight_layout()
plt.savefig(filename)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Scan over zetaUV
print 'Creating Zeta UV plot...'
pl.figure()
ax = pl.gca()
filename = test_plots + '/Test_' + ftag + '_ZetaUV_scan_zetaX_{:.0e}_Tmin_{:.1e}_Nalpha_{:.0e}_fpbh_{:.0e}.pdf'.format(zetaX, Tmin, Nalpha, fpbh)

zetaUVL = [15, 30, 50, 70, 90]

evalVec = []
for zuv in zetaUVL:
    if not tb_analysis or GlobalTb:
        for z in Zlist:
            evalVec.append([z, np.log10(fpbh), np.log10(zuv),
                            np.log10(zetaX), np.log10(Tmin), np.log10(Nalpha)])

    else:
        for kk in klist:
            evalVec.append([np.log10(kk), np.log10(fpbh), np.log10(zuv),
                            np.log10(zetaX), np.log10(Tmin), np.log10(Nalpha)])
vals = initPBH.eval_NN(evalVec=evalVec)

for i,zuv in enumerate(zetaUVL):
    inarray = np.asarray(evalVec)
    
    yvals = vals[inarray[:, -4] == np.log10(zuv)]
    xvals = inarray[inarray[:, -4] == np.log10(zuv)][:,0]
    if tb_analysis and not GlobalTb:
        xvals = np.power(10, xvals)
    yvals = yvals[np.argsort(xvals)]
    xvals.sort()
    pl.plot(xvals, yvals, lw=1, color=color_list[i], label=r'$\zeta_{UV} = $'+'{:.0f}'.format(zuv))
    if tb_analysis:
        if GlobalTb:
            tbfile = '../TbFiles/tb_file_mpbh_{:.0e}_fpbh_{:.0e}_zetaUV_{:.0f}_zetaX_{:.0e}_Tmin_{:.1e}_Nalpha_{:.0e}.dat'.format(Mpbh,fpbh,zuv,zetaX,Tmin,Nalpha)
        else:
            tbfile = '../TbFiles/tb_PowerSpectrum_mpbh_{:.0e}_fpbh_{:.0e}_zetaUV_{:.0f}_zetaX_{:.0e}_Tmin_{:.1e}_Nalpha_{:.0e}.dat'.format(Mpbh,fpbh,zuv,zetaX,Tmin,Nalpha)
    else:
        tbfile = '../XeFiles/xe_file_mpbh_{:.0e}_fpbh_{:.0e}_zetaUV_{:.0f}_zetaX_{:.0e}_Tmin_{:.1e}_Nalpha_{:.0e}.dat'.format(Mpbh,fpbh,zuv,zetaX,Tmin,Nalpha)
    if os.path.exists(tbfile):
        if not tb_analysis or GlobalTb:
            sim_data = np.loadtxt(tbfile)
        else:
            sim_data = power_spectrum_pull(tbfile, Zpower)
        pl.plot(sim_data[:,0], sim_data[:,1], 'o', ms=3, lw=1, color=color_list[i])

ax.axhline(y=0, xmin=0, xmax=50, color='k')
if tb_analysis:
    if GlobalTb:
        ax.set_ylabel(r'$\delta T_b$  [mK]', fontsize=fs)
        plt.xlim([6,30])
        ax.set_xlabel(r'z', fontsize=fs)
    else:
        ax.set_ylabel(r'$\Delta T_b$ ', fontsize=fs)
        ax.set_yscale('log')
        ax.set_xscale('log')
        plt.xlim([0.05, 2.])
        plt.ylim([0.1, 4e3])
        ax.set_xlabel(r'k', fontsize=fs)
else:
    ax.set_ylabel(r'$x_e$', fontsize=fs)
    plt.xlim([6,30])
    ax.set_xlabel(r'z', fontsize=fs)
plt.legend()
plt.tight_layout()
plt.savefig(filename)


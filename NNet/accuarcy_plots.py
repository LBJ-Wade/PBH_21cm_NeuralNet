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

rc('font',**{'family':'serif','serif':['Times','Palatino']})
rc('text', usetex=True)

mpl.rcParams['xtick.major.size']=8
mpl.rcParams['ytick.major.size']=8
mpl.rcParams['xtick.labelsize']=14
mpl.rcParams['ytick.labelsize']=14
fs=16

path = os.getcwd()

# Set global parameters and init NN
test_plots = os.getcwd() + '/Test_Plots/'
tb_analysis = True
GlobalTb = False
Mpbh = 100
Zlist = np.linspace(6, 35, 70)
klist = np.logspace(np.log10(0.05), np.log10(2), 70)
color_list = ['#9883E5', '#72A1E5', '#50C9CE', '#2E382E', 'b', 'r', 'g', 'k']

if tb_analysis:
    initPBH = Tb_PBH_Nnet(Mpbh, globalTb=GlobalTb)
    initPBH.main_nnet()
    ftag = 'Tb'
    if not GlobalTb:
        ftag += '_Power_'
else:
    initPBH = Xe_PBH_Nnet(Mpbh)
    initPBH.main_nnet()
    ftag = 'Xe'

fpbh = 1e-8
zetaUV = 50
zetaX = 2e56
Tmin = 5e4
Nalpha = 4e3

Zpower = 16.690


def power_spectrum_pull(file, zVal):
    data = np.loadtxt(file)
    indxs = np.isclose(data[:,0], zVal, atol=1e-2)
    data = data[indxs]
    return np.column_stack((data[:,1], data[:,2]))


# Scan over fpbh
print 'Creating fpbh plot...'
pl.figure()
ax = pl.gca()
filename = test_plots + '/Test_' + ftag + '_fpbh_scan.pdf'

fpbhL = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

evalVec = []
for f in fpbhL:
    if not tb_analysis or GlobalTb:
        for z in Zlist:
            evalVec.append([z, np.log10(f), np.log10(zetaUV), np.log10(zetaX), np.log10(Tmin), np.log10(Nalpha)])
    else:
        for kk in klist:
            evalVec.append([Zpower, kk, np.log10(f), np.log10(zetaUV), np.log10(zetaX), np.log10(Tmin), np.log10(Nalpha)])

vals = np.pow(10, initPBH.eval_NN(evalVec=evalVec))

for i,f in enumerate(fpbhL):
    if not tb_analysis or not GlobalTb:
        indx = 1
    else:
        indx = 2
    inarray = np.asarray(evalVec)
    yvals = vals[inarray[:, indx] == np.log10(f)]
    xvals = inarray[inarray[:, indx] == np.log10(f)][:,0]
    yvals = yvals[np.argsort(xvals)]
    xvals.sort()
    pl.plot(xvals, yvals, lw=1, color=color_list[i], label=r'$f_{pbh} = $'+'{:.0e}'.format(f))
    if tb_analysis:
        if GlobalTb:
            tbfile = '../TbFiles/tb_file_mpbh_{:.0e}_fpbh_{:.0e}_zetaUV_{:.0f}_zetaX_{:.0e}_Tmin_{:.1e}_Nalpha_{:.0e}.dat'.format(Mpbh,f,zetaUV,zetaX,Tmin,Nalpha)
        else:
            tbfile = 'TbFiles/tb_PowerSpectrum_mpbh_{:.0e}_fpbh_{:.0e}_zetaUV_{:.0f}_zetaX_{:.0e}_Tmin_{:.1e}_Nalpha_{:.0e}.dat'.format(Mpbh,f,zetaUV,zetaX,Tmin,Nalpha)
    else:
        tbfile = '../XeFiles/xe_file_mpbh_{:.0e}_fpbh_{:.0e}_zetaUV_{:.0f}_zetaX_{:.0e}_Tmin_{:.1e}_Nalpha_{:.0e}.dat'.format(Mpbh,f,zetaUV,zetaX,Tmin,Nalpha)
    if os.path.exists(tbfile):
        if not tb_analysis or GlobalTb:
            sim_data = np.loadtxt(tbfile)
        else:
            sim_data = power_spectrum_pull(tbfile, zVal)
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
        plt.xlim([0.5, 2.])
        ax.set_xlabel(r'k', fontsize=fs)
else:
    ax.set_ylabel(r'$x_e$', fontsize=fs)
    plt.xlim([6,30])
    ax.set_xlabel(r'z', fontsize=fs)
plt.legend()
plt.tight_layout()
plt.savefig(filename)

exit()

# Scan over Nalpha
print 'Creating Nalpha plot...'
pl.figure()
ax = pl.gca()
filename = test_plots + '/Test_' + ftag + '_Nalpha_scan.pdf'

NalphaL = [4e2, 1e3, 4e3, 1e4, 4e4]

evalVec = []
for na in NalphaL:
    for z in Zlist:
        evalVec.append([z, np.log10(fpbh), np.log10(zetaUV), np.log10(zetaX), np.log10(Tmin), np.log10(na)])
vals = initPBH.eval_NN(evalVec=evalVec)

for i,na in enumerate(NalphaL):
    inarray = np.asarray(evalVec)
    
    yvals = vals[inarray[:, -1] == np.log10(na)]
    xvals = inarray[inarray[:, -1] == np.log10(na)][:,0]
    yvals = yvals[np.argsort(xvals)]
    xvals.sort()
    pl.plot(xvals, yvals, lw=1, color=color_list[i], label=r'$N_\alpha = $'+'{:.1e}'.format(na))
    if tb_analysis:
        tbfile = '../TbFiles/tb_file_mpbh_{:.0e}_fpbh_{:.0e}_zetaUV_{:.0f}_zetaX_{:.0e}_Tmin_{:.1e}_Nalpha_{:.0e}.dat'.format(Mpbh,fpbh,zetaUV,zetaX,Tmin,na)
    else:
        tbfile = '../XeFiles/xe_file_mpbh_{:.0e}_fpbh_{:.0e}_zetaUV_{:.0f}_zetaX_{:.0e}_Tmin_{:.1e}_Nalpha_{:.0e}.dat'.format(Mpbh,fpbh,zetaUV,zetaX,Tmin,na)

    if os.path.exists(tbfile):
        #print 'File Exists.'
        sim_data = np.loadtxt(tbfile)
        pl.plot(sim_data[:,0], sim_data[:,1], 'o', ms=3, lw=1, color=color_list[i])

ax.axhline(y=0, xmin=0, xmax=50, color='k')
ax.set_xlabel(r'z', fontsize=fs)
if tb_analysis:
    ax.set_ylabel(r'$\delta T_b$  [mK]', fontsize=fs)
else:
    ax.set_ylabel(r'$x_e$', fontsize=fs)
plt.xlim([6,30])
plt.legend()
plt.tight_layout()
plt.savefig(filename)


# Scan over Tmin
print 'Creating Tmin plot...'
pl.figure()
ax = pl.gca()
filename = test_plots + '/Test_' + ftag + '_Tmin_scan.pdf'

TminL = [1e4, 2.5e4, 5e4, 7e4, 1e5]

evalVec = []
for tm in TminL:
    for z in Zlist:
        evalVec.append([z, np.log10(fpbh), np.log10(zetaUV), np.log10(zetaX), np.log10(tm), np.log10(Nalpha)])
vals = initPBH.eval_NN(evalVec=evalVec)

for i,tm in enumerate(TminL):
    inarray = np.asarray(evalVec)
    
    yvals = vals[inarray[:, -2] == np.log10(tm)]
    xvals = inarray[inarray[:, -2] == np.log10(tm)][:,0]
    yvals = yvals[np.argsort(xvals)]
    xvals.sort()
    pl.plot(xvals, yvals, lw=1, color=color_list[i], label=r'$T^{vir}_{min} = $'+'{:.1e}'.format(tm))
    if tb_analysis:
        tbfile = '../TbFiles/tb_file_mpbh_{:.0e}_fpbh_{:.0e}_zetaUV_{:.0f}_zetaX_{:.0e}_Tmin_{:.1e}_Nalpha_{:.0e}.dat'.format(Mpbh,fpbh,zetaUV,zetaX,tm,Nalpha)
    else:
        tbfile = '../XeFiles/xe_file_mpbh_{:.0e}_fpbh_{:.0e}_zetaUV_{:.0f}_zetaX_{:.0e}_Tmin_{:.1e}_Nalpha_{:.0e}.dat'.format(Mpbh,fpbh,zetaUV,zetaX,tm,Nalpha)

    if os.path.exists(tbfile):
        #print 'File Exists.'
        sim_data = np.loadtxt(tbfile)
        pl.plot(sim_data[:,0], sim_data[:,1], 'o', ms=3, lw=1, color=color_list[i])

ax.axhline(y=0, xmin=0, xmax=50, color='k')
ax.set_xlabel(r'z', fontsize=fs)
if tb_analysis:
    ax.set_ylabel(r'$\delta T_b$  [mK]', fontsize=fs)
else:
    ax.set_ylabel(r'$x_e$', fontsize=fs)
plt.xlim([6,30])
plt.legend()
plt.tight_layout()
plt.savefig(filename)

# Scan over zetaX
print 'Creating Zeta X plot...'
pl.figure()
ax = pl.gca()
filename = test_plots + '/Test_' + ftag + '_ZetaX_scan.pdf'

zetaXL = [2e55, 7e55, 2e56, 7e56, 2e57]

evalVec = []
for zx in zetaXL:
    for z in Zlist:
        evalVec.append([z, np.log10(fpbh), np.log10(zetaUV), np.log10(zx), np.log10(Tmin), np.log10(Nalpha)])
vals = initPBH.eval_NN(evalVec=evalVec)

for i,zx in enumerate(zetaXL):
    inarray = np.asarray(evalVec)
    
    yvals = vals[inarray[:, -3] == np.log10(zx)]
    xvals = inarray[inarray[:, -3] == np.log10(zx)][:,0]
    yvals = yvals[np.argsort(xvals)]
    xvals.sort()
    pl.plot(xvals, yvals, lw=1, color=color_list[i], label=r'$\zeta_{X} = $'+'{:.1e}'.format(zx))
    if tb_analysis:
        tbfile = '../TbFiles/tb_file_mpbh_{:.0e}_fpbh_{:.0e}_zetaUV_{:.0f}_zetaX_{:.0e}_Tmin_{:.1e}_Nalpha_{:.0e}.dat'.format(Mpbh,fpbh,zetaUV,zx,Tmin,Nalpha)
    else:
        tbfile = '../XeFiles/xe_file_mpbh_{:.0e}_fpbh_{:.0e}_zetaUV_{:.0f}_zetaX_{:.0e}_Tmin_{:.1e}_Nalpha_{:.0e}.dat'.format(Mpbh,fpbh,zetaUV,zx,Tmin,Nalpha)

    if os.path.exists(tbfile):
        #print 'File Exists.'
        sim_data = np.loadtxt(tbfile)
        pl.plot(sim_data[:,0], sim_data[:,1], 'o', ms=3, lw=1, color=color_list[i])

ax.axhline(y=0, xmin=0, xmax=50, color='k')
ax.set_xlabel(r'z', fontsize=fs)
if tb_analysis:
    ax.set_ylabel(r'$\delta T_b$  [mK]', fontsize=fs)
else:
    ax.set_ylabel(r'$x_e$', fontsize=fs)
plt.xlim([6,30])
plt.legend()
plt.tight_layout()
plt.savefig(filename)


# Scan over zetaUV
print 'Creating Zeta UV plot...'
pl.figure()
ax = pl.gca()
filename = test_plots + '/Test_' + ftag + '_ZetaUV_scan.pdf'

zetaUVL = [10, 30, 50, 70, 90]

evalVec = []
for zuv in zetaUVL:
    for z in Zlist:
        evalVec.append([z, np.log10(fpbh), np.log10(zuv), np.log10(zetaX), np.log10(Tmin), np.log10(Nalpha)])
vals = initPBH.eval_NN(evalVec=evalVec)

for i,zuv in enumerate(zetaUVL):
    inarray = np.asarray(evalVec)
    
    yvals = vals[inarray[:, -4] == np.log10(zuv)]
    xvals = inarray[inarray[:, -4] == np.log10(zuv)][:,0]
    yvals = yvals[np.argsort(xvals)]
    xvals.sort()
    pl.plot(xvals, yvals, lw=1, color=color_list[i], label=r'$\zeta_{UV} = $'+'{:.0f}'.format(zuv))
    if tb_analysis:
        tbfile = '../TbFiles/tb_file_mpbh_{:.0e}_fpbh_{:.0e}_zetaUV_{:.0f}_zetaX_{:.0e}_Tmin_{:.1e}_Nalpha_{:.0e}.dat'.format(Mpbh,fpbh,zuv,zetaX,Tmin,Nalpha)
    else:
        tbfile = '../XeFiles/xe_file_mpbh_{:.0e}_fpbh_{:.0e}_zetaUV_{:.0f}_zetaX_{:.0e}_Tmin_{:.1e}_Nalpha_{:.0e}.dat'.format(Mpbh,fpbh,zuv,zetaX,Tmin,Nalpha)

    if os.path.exists(tbfile):
        #print 'File Exists.'
        sim_data = np.loadtxt(tbfile)
        pl.plot(sim_data[:,0], sim_data[:,1], 'o', ms=3, lw=1, color=color_list[i])

ax.axhline(y=0, xmin=0, xmax=50, color='k')
ax.set_xlabel(r'z', fontsize=fs)
if tb_analysis:
    ax.set_ylabel(r'$\delta T_b$  [mK]', fontsize=fs)
else:
    ax.set_ylabel(r'$x_e$', fontsize=fs)
plt.xlim([6,30])
plt.legend()
plt.tight_layout()
plt.savefig(filename)



import os
import numpy as np
from scipy.interpolate import interp1d

Mpbh = 100
arrayN = 'hera331'
fileN = 'Chi2_Fits_' + arrayN + '_TbPower_Mpbh_{:.0f}'.format(Mpbh) + '_ModerateSense.dat'

chi2Vals = np.loadtxt(fileN)
fpbhVals = np.unique(chi2Vals[:,0])
limArr = np.zeros((len(fpbhVals), 2))

for i,fpbh in enumerate(fpbhVals):
    data_subset = chi2Vals[chi2Vals[:,0] == fpbh]
    limArr[i,0] = fpbh
    limArr[i,1] = np.min(data_subset[:,-1])

    
print limArr

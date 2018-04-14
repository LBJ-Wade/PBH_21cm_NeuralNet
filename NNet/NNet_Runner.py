import os
from Mpbh_Nnet import *

#Runner for training and testing. Not intended for integration into code. 
Mpbh = 100.

# Train NNet
Train = True
KeepTraining = False

# Load and Evaluate NNet
Eval = False

#Global Tb or power spectrum
GlobalTb = True


redshift_1 = 10.5
redshift_2 = 17.5
fpbh = np.log10(1e-8)
zetaUV = np.log10(50)
zetaX = np.log10(2e56)
Tmin = np.log10(5e4)
Nalpha = np.log10(4e3)
vec_in = [[redshift_1, fpbh, zetaUV, zetaX, Tmin, Nalpha], [redshift_2, fpbh, zetaUV, zetaX, Tmin, Nalpha]]
if not GlobalTb:
    k = 0.1
    vec_in = [[redshift, k, fpbh, zetaUV, zetaX, Tmin, Nalpha]]
#PBH_Nnet(Mpbh, globalTb=GlobalTb).main_nnet(train_nnet=Train, eval_nnet=Eval, keep_training=KeepTraining, evalVec=vec_in)
init_pbh = PBH_Nnet(Mpbh, globalTb=GlobalTb)
init_pbh.main_nnet()
if Train:
    init_pbh.train_NN(vec_in, keep_training=KeepTraining)
if Eval:
    init_pbh.eval_NN(vec_in)

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 18:09:51 2023

@author: Krohalev_OD

testing new interpolation
"""

import hibplib as hb
import numpy as np
import matplotlib.pyplot as plt

# %% Load Magnetic Field

pf_coils = hb.import_PFcoils('PFCoils.dat')
PF_dict = hb.import_PFcur('{}MA_sn.txt'.format(int(abs(1.0))), pf_coils)
dirname = 'magfield'
B_old = hb.read_B(1.0, 1.0, PF_dict, dirname=dirname, plot=False)
B_new = hb.read_B_new(1.0, 1.0, PF_dict, dirname=dirname, plot=False)

#%% testing on random r

log = []
accuracy = 0.01
string = ''
z = 0.
y = 0.
xx = np.linspace(1.1, 5.4, 10000)

def BB(xx, _return_B, B_data): 
    result = np.zeros( (3, len(xx)) )
    
    for i, x in enumerate(xx): 
        r = np.array([x, y, z])
        #B = hb.return_B_new(r, B_new)
        B = _return_B(r, B_data)
        if B.shape == (1, 3): 
            result[:, i] = B[0]
        else: 
            result[:, i] = B
    return result

# yy = BB(xx, hb.return_B_new, B_new)
# plt.plot(xx, yy[1]) 

# #yy1 = BB(xx, hb.return_B, B_old)
# #plt.plot(xx, yy1[1]) 

# raise Exception

for i in range(10000):
    r = np.random.rand(1, 3)[0] + np.array([1.1, -0.5, -0.5])
    old_B = hb.return_B(r, B_old)
    new_B = hb.return_B_new(r, B_new)
    if np.any(abs(new_B - old_B) > accuracy):
        string = 'B are NOT the same, new B = ' + str(new_B) + ', old B = ' + \
            str(old_B) + ', r = ' + str(r)
        log.append(string)
        print(string)
    else:
        string = 'B are the same, new B = ' + str(new_B) + ', old B = ' + \
            str(old_B) + ', r = ' + str(r)
        print(string)


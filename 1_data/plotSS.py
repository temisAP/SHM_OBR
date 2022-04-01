#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os
import sys

sys.path.append('../0_sources/PYTHON')

from SPECTRAL_SHIFT.read_obr import multi_read_obr
from SPECTRAL_SHIFT.Spectral_Shift import global_spectral_shift
from UTILS.utils import find_all_OBR
from PLOTS.plots import temperature_plot
import numpy as np



""" Path to data and REF """

folder = 'CM2/0_OBR'
path_to_data = f'/mnt/sda/0_Andres/1_Universidad/Beca_SHM/98_data/0_CALIBRACION/{folder}'
files = ['19_grados','30_grados','172_grados']

""" Compute and plot ss """

# Parameters
REF = '19_grados'
limit1 = 0
limit2 = 5.5
delta = 300
window = 1000

# Get reference
files.remove(REF)
f,z,Data = multi_read_obr([REF],path_to_data,limit1=limit1,limit2=limit2)
Data_REF = Data[REF]

# Get spectralshifts
spectralshifts = list()
for i,file in enumerate(files):
    f,z,Data = multi_read_obr([file],path_to_data,limit1=limit1,limit2=limit2)
    spectralshift = global_spectral_shift(Data_REF[0],Data[file][0],f,delta=delta,window=window)
    spectralshifts.append(spectralshift)

z = np.linspace(z[0],z[-1],len(spectralshift))

# Plot

import  matplotlib.pyplot as plt
for spectralshift,file in zip(spectralshifts,files):
    plt.plot(z,spectralshift,label = file)

plt.grid()

"""
# Read files

folder = 'Temperature'
path_to_data = f'/mnt/sda/0_Andres/1_Universidad/Beca_SHM/98_data/0_CALIBRACION/CMT/{folder}'
files = ['ref','ini','fin']


# Parameters
REF = 'ref'

# Get reference
files.remove(REF)
f,z,Data = multi_read_obr([REF],path_to_data,limit1=limit1,limit2=limit2)
Data_REF = Data[REF]

# Get spectralshifts
spectralshifts = list()
for i,file in enumerate(files):
    f,z,Data = multi_read_obr([file],path_to_data,limit1=limit1,limit2=limit2)
    spectralshift = global_spectral_shift(Data_REF[0],Data[file][0],f,delta=delta,window=window)
    spectralshifts.append(spectralshift)

z = np.linspace(z[0],z[-1],len(spectralshift))

# Plot
for spectralshift,file in zip(spectralshifts,files):
    plt.plot(z,spectralshift,label=file)
"""

plt.legend()
plt.show()


# In[ ]:

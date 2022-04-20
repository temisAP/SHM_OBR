import sys
import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../0_sources/PYTHON'))
from UTILS.read_obr import multi_read_obr
from SIGNAL.Spectral_Shift import global_spectral_shift
from UTILS.utils import get_all_files, find_index
from PLOTS.plots import temperature_plot


""" Path to data and REF """

folder = 'CT2'
path_to_data = f'/mnt/sda/0_Andres/1_Universidad/Beca_SHM/98_data/0_CALIBRACION/{folder}'
files = get_all_files(path_to_data)
files = files[0:-1:10]
#files = [file for file in files if '22_grados' in file]

#REF = '0_mm_22_grados'
REF = '19_grados'
files.remove(REF)

# Parameters
limit1 = 2.3
limit2 = 3.5
delta = 300
window = 1000

# Get reference
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
temperature_plot(z,spectralshifts,files,REF)
plt.show()

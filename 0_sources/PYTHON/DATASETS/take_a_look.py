import sys
import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from UTILS.read_obr import multi_read_obr
from SIGNAL.Spectral_Shift import global_spectral_shift


def take_a_look(self,REF,files,limit1=0, limit2 = 20, delta = 300, window = 1000, val='ss',plot=True):

    # Get reference
    f,z,Data = multi_read_obr([REF],os.path.join(self.path,self.folders['0_OBR']),limit1=limit1,limit2=limit2)
    refData = Data[REF]

    val_distributions = list()
    for file in files:
        f,z,Data = multi_read_obr([file],os.path.join(self.path,self.folders['0_OBR']),limit1=limit1,limit2=limit2)
        if val == 'ss':
            spectralshift = global_spectral_shift(refData[0],Data[file][0],f,delta=delta,window=window)
            val_distribution = spectralshift
            ylabel = r'$-\frac{\Delta \nu}{\bar{\nu}}$'
        elif val == 'temp':
            T,E = self.sensor(Data[file],refData,f,delta=delta,window=window,display = False)
            val_distribution = T
            ylabel = r'$\Delta T$[K]'
        elif val == 'def':
            T,E = self.sensor(Data[file],refData,f,delta=delta,window=window,display = False)
            val_distribution = E
            ylabel = r'$\Delta \mu \varepsilon$'

        val_distributions.append(val_distribution)

    z = np.linspace(z[0],z[-1],len(val_distribution))


    # Plot
    if plot:
        plt.figure()
        for file, val_distribution in zip(files, val_distributions):
            plt.plot(z,val_distribution, label=f'{file}')
        plt.ylabel(ylabel,fontsize=20,labelpad=15).set_rotation(0) if val == 'ss' else plt.ylabel(ylabel,labelpad=5).set_rotation(0)
        plt.xlabel('z [m]')
        plt.legend()
        plt.grid()
        plt.show()

    return val_distributions

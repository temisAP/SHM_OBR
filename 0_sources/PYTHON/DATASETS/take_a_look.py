import sys
import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from UTILS.read_obr import multi_read_obr
from SIGNAL.Spectral_Shift import global_spectral_shift
from SIGNAL.Birefringence import global_birefringence

def take_a_look(self,REF,files,limit1=0, limit2 = 20, delta = 300, window = 1000, val='ss',plot=True, common_origin=False):

    # Get reference
    if isinstance(REF,list):
        refData = list()
        for i,r in enumerate(REF):
            f,z,Data = multi_read_obr([r],os.path.join(self.path,self.folders['0_OBR']),limit1=limit1[i],limit2=limit2[i])
            refData.append(Data[r])
        reftype = 'multiple'
    else:
        f,z,Data = multi_read_obr([REF],os.path.join(self.path,self.folders['0_OBR']),limit1=limit1,limit2=limit2)
        refData = Data[REF]
        reftype = 'single'


    # Get distributions
    val_distributions = list()
    z_distributions = list()
    for i,file in enumerate(files):
        if reftype == 'single':
            r  = refData
            l1 = limit1
            l2 = limit2
        elif reftype == 'multiple':
            r  = refData[i]
            l1 = limit1[i]
            l2 = limit2[i]

        f,z,Data = multi_read_obr([file],os.path.join(self.path,self.folders['0_OBR']),limit1=l1,limit2=l2)
        if val == 'ss':
            spectralshift = global_spectral_shift(r[0],Data[file][0],f,delta=delta,window=window)
            val_distribution = spectralshift
            ylabel = r'$-\frac{\Delta \nu}{\bar{\nu}}$'
        elif val == 'brf':
            brf = birefringence(Data[file],f,delta=delta,window=window)
            val_distribution = brf
            ylabel = r'$\theta$'
        elif val == 'temp':
            T,E = self.sensor(Data[file],r,f,delta=delta,window=window,display = False)
            val_distribution = T
            ylabel = r'$\Delta T$[K]'
        elif val == 'def':
            #T,E = self.sensor(Data[file],r,f,delta=delta,window=window,display = False)
            spectralshift = global_spectral_shift(r[0],Data[file][0],f,delta=delta,window=window)
            E = spectralshift * 189394.17022471415 * -6.6680 * -8.80840637e2/119.88246527477283
            val_distribution = E
            ylabel = r'$\Delta \mu \varepsilon$'

        val_distributions.append(val_distribution)
        z_distributions.append(np.linspace(z[0],z[-1],len(val_distribution)))


    # Plot
    if plot:
        plt.figure()
        for file, val, z in zip(files, val_distributions, z_distributions):
            lbl = file.replace('_',' ')
            z = z-z[0] if common_origin else z
            #plt.plot(z,val,'o',markersize=0.75,label=f'{lbl}')
            plt.plot(z,val,label=f'{lbl}')
        plt.ylabel(ylabel,fontsize=20,labelpad=15).set_rotation(0) if val == 'ss' else plt.ylabel(ylabel,fontsize=10,labelpad=8).set_rotation(0)
        plt.xlabel('z [m]')

        """
        lgnd = plt.legend(numpoints=1, fontsize=10)

        #change the marker size manually for both lines
        lgnd.legendHandles[0]._legmarker.set_markersize(6)
        lgnd.legendHandles[1]._legmarker.set_markersize(6)
        lgnd.legendHandles[2]._legmarker.set_markersize(6)
        """
        plt.legend()
        plt.grid()
        plt.show()

    return val_distributions

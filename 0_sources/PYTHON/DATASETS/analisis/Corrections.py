import sys
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from UTILS.read_obr import multi_read_obr
from SIGNAL.Spectral_Shift import global_spectral_shift
from SIGNAL.NonlinearCorrections import NonlinearCorrections

def Corrections(self,REF,files,limit1=0, limit2 = 20, delta = 300, window = 1000, val='ss',plot=True, common_origin=False):

    # Manage colormap
    if len(files) <= 10:
        colormap = mpl.cm.get_cmap('tab10')
    else:
        colormap = mpl.cm.get_cmap('rainbow')


    # Get reference
    if isinstance(REF,list):
        refData = list()
        crefData = list()
        for i,r in enumerate(REF):
            f,z,Data = multi_read_obr([r],os.path.join(self.path,self.folders['0_OBR']),limit1=limit1[i],limit2=limit2[i])
            refData.append(Data[r])
            crefData.append(Data_correction(Data[r]))
        reftype = 'multiple'
    else:
        f,z,Data = multi_read_obr([REF],os.path.join(self.path,self.folders['0_OBR']),limit1=limit1,limit2=limit2)
        refData  = Data[REF]
        crefData = Data_correction(Data[REF])
        reftype  = 'single'


    # Get distributions
    val_distributions   = list()
    cval_distributions  = list()
    z_distributions     = list()

    Data  = dict().fromkeys(files)
    cData = dict().fromkeys(files)


    for i,file in enumerate(files):

        # Set limits and references
        if reftype == 'single':
            r  = refData
            cr = crefData
            l1 = limit1
            l2 = limit2
        elif reftype == 'multiple':
            r  = refData[i]
            cr = crefData[i]
            l1 = limit1[i]
            l2 = limit2[i]

        # Read raw data and correct it
        f,z,Signals = multi_read_obr([file],os.path.join(self.path,self.folders['0_OBR']),limit1=l1,limit2=l2)
        Data[file]  = Signals[file]
        cData[file] = Data_correction(Data[file])


        if val == 'ss':
            val_distribution  = global_spectral_shift(r[0],Data[file][0],f,delta=delta,window=window)
            cval_distribution = global_spectral_shift(cr[0],cData[file][0],f,delta=delta,window=window)
            set_ylabel            = r'$-\frac{\Delta \nu}{\bar{\nu}}$'
        elif val == 'brf':
            val_distribution  = birefringence(Data[file],f,delta=delta,window=window)
            cval_distribution = birefringence(cData[file],f,delta=delta,window=window)
            set_ylabel            = r'$\theta$'
        elif val == 'temp':
            T,E = self.sensor(Data[file],r,f,delta=delta,window=window,display = False)
            val_distribution = T
            set_ylabel = r'$\Delta T$[K]'
        elif val == 'def':
            #T,E = self.sensor(Data[file],r,f,delta=delta,window=window,display = False)
            spectralshift = global_spectral_shift(r[0],Data[file][0],f,delta=delta,window=window)
            E = spectralshift * 189394.17022471415 * -6.6680 * -8.80840637e2/119.88246527477283
            val_distribution = E
            set_ylabel = r'$\Delta \mu \varepsilon$'

        val_distributions.append(val_distribution)
        cval_distributions.append(cval_distribution)
        z_distributions.append(np.linspace(z[0],z[-1],len(val_distribution)))


    # Plot
    i = 0
    if plot:
        fig, ax = plt.subplots(3)
        for file, val, cval, z in zip(files, val_distributions, cval_distributions , z_distributions):
            lbl = file.replace('_',' ')

            c =colormap(i);i+=1
            markersize = 6
            marker_original = '-v'
            marker_corrected = '-^'

            z = z-z[0] if common_origin else z
            z_raw = np.linspace(z[0],z[-1],len(Data[file][0]))

            # Signal comparison (P)
            ax[0].plot(z_raw,np.abs(Data[file][0]),marker_original,color = c, markersize=markersize,label=f'{lbl} (original)')
            ax[0].plot(z_raw,np.abs(cData[file][0]),marker_corrected,color = c, markersize=markersize,label=f'{lbl} (corrected)')
            # Signal comparisson (S)
            ax[1].plot(z_raw,np.abs(Data[file][1]),marker_original,color = c, markersize=markersize,label=f'{lbl} (original)')
            ax[1].plot(z_raw,np.abs(cData[file][1]),marker_corrected,color = c, markersize=markersize,label=f'{lbl} (corrected)')
            # Val comparisson
            ax[2].plot(z,val,marker_original,color = c, markersize=markersize,label=f'{lbl} (original)')
            ax[2].plot(z,cval,marker_corrected,color = c, markersize=markersize,label=f'{lbl} (corrected)')

        ax[0].set_ylabel('P',fontsize=20,labelpad=15).set_rotation(0)
        ax[0].set_xlabel('z [m]')
        ax[1].set_ylabel('S',fontsize=20,labelpad=15).set_rotation(0)
        ax[1].set_xlabel('z [m]')
        ax[2].set_ylabel(set_ylabel,fontsize=20,labelpad=15).set_rotation(0) if val == 'ss' else ax[2].set_ylabel(set_ylabel,fontsize=10,labelpad=8).set_rotation(0)
        ax[2].set_xlabel('z [m]')

        ax[0].get_shared_x_axes().join(ax[0], ax[1])
        ax[0].get_shared_x_axes().join(ax[0], ax[2])

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.figlegend(by_label.values(), by_label.keys(),loc='lower center',ncol=len(files),fancybox=False, shadow=False)

        ax[0].grid()
        ax[1].grid()
        ax[2].grid()
        plt.show()

    return val_distributions


def Data_correction(Data):
    Data[0] = NonlinearCorrections(Data[0])
    Data[1] = NonlinearCorrections(Data[1])
    return Data

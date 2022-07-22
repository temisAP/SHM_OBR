import matplotlib.pyplot as plt
import librosa
import librosa.display
from .Z9_utils import arr2librosa, a_plot, create_data_and_ylabels
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from SIGNAL.Spectral_Shift import global_spectral_shift
from UTILS.utils import check_memory


def Spectral_Shift(samples,magnitude='p-s',delta=200, window=1000):

    """ Function to show the spectral shift of a choosen magnitude along the fiber for the choosen samples

            : param samples (list of strings): names of the files used as example

            : optional magnitude (string): maginitude which will be compared

                    available options -> "p-s":  p and s polarization states
                                      -> "St" :  Stokes vector
                                      -> "S_0":  first component of the Stokes vector
                                      -> "S_1":  second component of the Stokes vector
                                      -> "S_2":  third component of the Stokes vector
                                      -> "S_3":  fourth component of the Stokes vector

            : optional delta    (int):
            : optional window   (int):

    """

    if magnitude == 'p-s':
        states = [rf'$P_0 \star P_1$',rf'$S_0 \star S_1$']
        ref_data = [[0],[0]]
    elif magnitude == 'St':
        states = [rf'$\vec{{S}}$']
        ref_data = [[0]]
    elif 'S_' in magnitude:
        X  = int(magnitude.split('_')[1])
        states = [rf'$S_{X}$']
        ref_data = [[0]]


    components = ['Spectral shift']

    data, ylabels = create_data_and_ylabels(samples.keys(),states,components)

    for i, sample, sample_key in zip(range(len(samples.keys())), samples.values(), samples.keys()):

        Temperature = sample.temperature
        Flecha = sample.flecha

        # Set reference
        if i == 0:
            if len(ref_data) == 2:
                ref_data[0] = sample.Data[0]
                ref_data[1] = sample.Data[1]
            elif len(ref_data) == 1:
                if 'S_' in magnitude:
                    ref_data[0] = sample.Data[2][X]
                elif magnitude == 'St':
                    ref_data[0] = sample.Data[2]

        # Compute spectral shift for desired values
        for j, state in enumerate(states):

            if len(ref_data) == 2:
                ss = global_spectral_shift(ref_data[j],sample.Data[j],sample.f,delta=delta,window=window)*1e6

            elif len(ref_data) == 1:
                if 'S_' in magnitude:
                    ss = global_spectral_shift(ref_data[0],sample.Data[2][X],sample.f,delta=delta,window=window)*1e6
                elif magnitude == 'St':
                    ss = global_spectral_shift(ref_data[0],sample.Data[2],sample.f,delta=delta,window=window)*1e6

            z  = np.linspace(0,sample.z[-1]-sample.z[0],len(ss))

            data[sample_key][state]['Spectral shift'] = [z,ss]

            ylabels[sample_key][state] =r'$-\frac{\Delta \nu}{\bar{\nu}} \: \cdot 10^6 $'+'\n'+rf'$T = {Temperature}\: Cº$'+'\n'+ rf'$\delta = {Flecha}\: mm$'

            j+=1

    # Make the plot
    a_plot(data,ylabels,type='',alpha = 1,linestyle='-',linewidth = 2)


    plt.show()

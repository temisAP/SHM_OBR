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


def Spectral_Shift(samples):

    sample_keys = samples.keys()
    states = [rf'$P_0 \star P_1$',rf'$S_0 \star S_1$']
    components = ['Spectral shift']

    data, ylabels = create_data_and_ylabels(sample_keys,states,components)

    ref_data = [[0],[0]]

    for i, sample, sample_key in zip(range(len(samples.keys())), samples.values(), samples.keys()):

        Temperature = sample.temperature
        Flecha = sample.flecha
        sr = int(1/(sample.z[1]-sample.z[0]))

        if i == 0:
            ref_data[0] = sample.Data[0]
            ref_data[1] = sample.Data[1]

        for j, state in enumerate(states):
            j+=1

            ss = global_spectral_shift(ref_data[j-1],sample.Data[j],sample.f,delta=200,window=250)*1e6
            z  = np.linspace(0,sample.z[-1]-sample.z[0],len(ss))

            data[sample_key][state]['Spectral shift'] = [z,ss]

            ylabels[sample_key][state] =r'$-\frac{\Delta \nu}{\bar{\nu}} \: \cdot 10^6 $'+'\n'+rf'$T = {Temperature}\: Cº$'+'\n'+ rf'$\delta = {Flecha}\: mm$'


    # Make the plot
    a_plot(data,ylabels,type='',alpha = 1,linestyle='-',linewidth = 2)


    plt.show()

import matplotlib.pyplot as plt
import librosa
import librosa.display
from .Z9_utils import arr2librosa, a_plot, create_data_and_ylabels
import numpy as np
from math import atan2


def Representation(samples):

    sample_keys = samples.keys()
    states = [r'$P(z)$',r'$S(z)$']
    components = ['Real', 'Imaginary']

    data, ylabels = create_data_and_ylabels(sample_keys,states,components)



    for i, sample, sample_key in zip(range(len(samples.keys())), samples.values(), samples.keys()):

        Temperature = sample.temperature
        Flecha = sample.flecha
        sr = int(1/(sample.z[1]-sample.z[0]))

        if i == 0:
            print('Points pad:',(sample.z[1]-sample.z[0])*1e6,'μm' )
            print('Sample rate:',sr,'1/m')

        for j, state in enumerate(states):

            wave, new_sr = arr2librosa(sample.Data[j],sr)

            data[sample_key][state]['Real']         = [wave.real, new_sr]
            data[sample_key][state]['Imaginary']    = [wave.imag, new_sr]

            ylabels[sample_key][state] = rf'$T = {Temperature}\: Cº$'+'\n'+ rf'$\delta = {Flecha}\: mm$'


    # Make the plot
    a_plot(data,ylabels)


    plt.show()

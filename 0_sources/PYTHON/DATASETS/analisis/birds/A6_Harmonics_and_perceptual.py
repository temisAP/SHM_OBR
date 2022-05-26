import matplotlib.pyplot as plt
import librosa
import librosa.display
from .Z9_utils import arr2librosa, a_plot, create_data_and_ylabels
import numpy as np

def Harmonics_and_perceptual(samples):

    sample_keys = samples.keys()
    states = [r'$P(z)$',r'$S(z)$']
    components = ['Real, harmonics', 'Imaginary, harmonics','Real, perceptual', 'Imaginary, perceptual']

    data, ylabels = create_data_and_ylabels(sample_keys,states,components)

    for i, sample, sample_key in zip(range(len(samples.keys())), samples.values(), samples.keys()):

        Temperature = sample.temperature
        Flecha = sample.flecha
        sr = int(1/(sample.z[1]-sample.z[0]))

        for j, state in enumerate(states):
            j+=1

            # np.array to librosa stuff
            wave, new_sr = arr2librosa(sample.Data[j],sr)

            # Harmonics and perceptual
            wave.real_harm, wave.real_perc = librosa.effects.hpss(wave.real)
            wave.imag_harm, wave.imag_perc = librosa.effects.hpss(wave.imag)


            data[sample_key][state]['Real, harmonics']         = [wave.real_harm, new_sr]
            data[sample_key][state]['Imaginary, harmonics']    = [wave.imag_harm, new_sr]
            data[sample_key][state]['Real, perceptual']        = [wave.real_perc, new_sr]
            data[sample_key][state]['Imaginary, perceptual']   = [wave.imag_perc, new_sr]

            ylabels[sample_key][state] = rf'$T = {Temperature}\: Cº$'+'\n'+ rf'$\delta = {Flecha}\: mm$'


    # Make the plot
    a_plot(data,ylabels)

    plt.subplots_adjust(top=0.954,
                            bottom=0.091,
                            left=0.091,
                            right=0.985,
                            hspace=0.0,
                            wspace=0.0)

    plt.show()

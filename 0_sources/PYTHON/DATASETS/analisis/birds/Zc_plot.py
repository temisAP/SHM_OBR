import librosa
from scipy.io.wavfile import write
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from .Z9_utils import custom_stft, correlation2D, a_plot, create_data_and_ylabels

def c_plot(data,ylabels,n_fft = 1000, hop_length = 200,
            type='librosa', alpha = 1,  linewidth = 0.5,linestyle = '-o'):

    """ Plot comprissons between data

        :param data: dict with the following structure

                data[sample][state] = {'label1': line1,'label2': line2, ... ,'labelN',lineN}

    """

    samples     = list(data.keys())
    states      = list(data[samples[0]].keys())
    components  = list(data[samples[0]][states[0]])

    for sample in samples:
        for state in states:
            data[sample][state] = {k: v for k, v in data[sample][state].items() if v} # Delete dict elements if empty
    components  = list(data[samples[0]][states[0]])

    a_data, a_ylabels   = create_data_and_ylabels(samples,states,components) # Difference
    b_data, b_ylabels   = create_data_and_ylabels(samples,states,components) # Spectrogram
    c_data, c_ylabels   = create_data_and_ylabels(samples,states,components) # Spectrogram 2D corr
    d_data, d_ylabels   = create_data_and_ylabels(samples,states,components) # Spectrogram 2D corr comp

    ref_y, _       = create_data_and_ylabels(samples,states,components)
    ref_S, _       = create_data_and_ylabels(samples,states,components)
    base_corr, _   = create_data_and_ylabels(samples,states,components)


    for i, sample in enumerate(samples):

        for j, state in enumerate(states):

            for magnitude in components:

                if type == 'librosa':
                    y = data[sample][state][magnitude][0]*1e6
                    new_sr = data[sample][state][magnitude][1]

                    S = custom_stft(y, window=n_fft,delta=hop_length)

                    # Reference states
                    if i == 0:
                        ref_sample = sample
                        ref_y[ref_sample][state][magnitude]        = y
                        ref_S[ref_sample][state][magnitude]        = S
                        base_corr[ref_sample][state][magnitude]    = correlation2D(ref_S[ref_sample][state][magnitude],ref_S[ref_sample][state][magnitude],axis=1)


                    # Difference
                    a_data[sample][state][magnitude] = [y-ref_y[ref_sample][state][magnitude],new_sr]

                    # Spectrogram
                    b_data[sample][state][magnitude] = [S, new_sr]

                    # Correlation between signals
                    c_data[sample][state][magnitude] = [correlation2D(ref_S[ref_sample][state][magnitude],S,axis=1), new_sr]

                    # Difference between correlation
                    d_data[sample][state][magnitude] = [correlation2D(ref_S[ref_sample][state][magnitude],S,axis=1)-base_corr[ref_sample][state][magnitude], new_sr]



                else:
                    y = data[sample][state][magnitude][1]*1e6
                    t = data[sample][state][magnitude][0]
                    new_sr = len(y)/len(t)

                    # Reference states
                    if i == 0:
                        ref_sample = sample
                        ref_y[ref_sample][state][magnitude]     = y

                    # diffence between reference states
                    a_data[sample][state][magnitude] = [t,y-ref_y[ref_sample][state][magnitude]]


                a_ylabels[sample][state] = b_ylabels[sample][state] = c_ylabels[sample][state] = d_ylabels[sample][state] = '1e-6\n\n'+ylabels[sample][state]


    # Difference
    a_plot(a_data,a_ylabels,type=type,linewidth=1)
    plt.get_current_fig_manager().canvas.set_window_title('Difference')

    if type == 'librosa':

        # Spectrogram
        a_plot(b_data,b_ylabels,hop_length = hop_length, x_axis = 'time', y_axis = 'linear', dB = False)
        plt.get_current_fig_manager().canvas.set_window_title('Spectrogram')

        # Spectrogram Correlation
        a_plot(c_data,c_ylabels,hop_length = hop_length, x_axis = 'time', y_axis = 'linear',dB = False)
        plt.get_current_fig_manager().canvas.set_window_title('Correlation')

        # Spectrogram Difference between correlations
        a_plot(d_data,d_ylabels,hop_length = hop_length, x_axis = 'time', y_axis = 'linear',dB = False)
        plt.get_current_fig_manager().canvas.set_window_title('Correlation comparisson')

import librosa
from scipy.io.wavfile import write
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def arr2librosa(array,sr):

    """ Function to create a .wav file from a numpy array and later open it with librosa

            : param array (np.array)  : a numpy array
            : param sr    (int)       : sample rate

            If array is compound any complex number

                : return wave (a_wave object): an object which contains

                                                    wave.real : the real part of the array
                                                    wave.imag : the imaginary part of the array
                                                    wave.mod  : the module of the array
                                                    wave.phase : the phase of the array

            else


                : return wave (librosa array): the original numpy array but in the librosa way

            : return new_sr (int) : because, for some reason is different from the original

            """

    if any(np.iscomplex(array)):

        class a_wave(object):
            def __init__(self):
                self.real   = ''
                self.imag   = ''
                self.mod    = ''
                self.phase  = ''

        wave = a_wave()

        write('temp.wav', sr, array.real)
        wave.real, new_sr  = librosa.load('temp.wav')
        os.remove('temp.wav')

        write('temp.wav', sr, array.imag)
        wave.imag, new_sr = librosa.load('temp.wav')
        os.remove('temp.wav')

        write('temp.wav', sr, np.abs(array))
        wave.mod, new_sr  = librosa.load('temp.wav')
        os.remove('temp.wav')

        write('temp.wav', sr, np.angle(array))
        wave.phase, new_sr = librosa.load('temp.wav')
        os.remove('temp.wav')

    else:

        write('temp.wav', sr, array)
        wave, new_sr = librosa.load('temp.wav')
        os.remove('temp.wav')

    return wave, new_sr

def create_data_and_ylabels(sample_keys,states,components):

    data = dict.fromkeys(sample_keys)
    for sample in data.keys():
        data[sample] = dict.fromkeys(states)
        for state in data[sample].keys():
            data[sample][state] = dict.fromkeys(components)

    import copy
    ylabels = copy.deepcopy(data)
    print('ylabels is data') if ylabels is data else False

    return data, ylabels

def a_plot(data,ylabels,
            type='librosa', alpha = 0.6,  linewidth = 0.5,
            cmap='jet',x_axis = 'time', y_axis = 'log', hop_length=None, dB = False):

    """ Plot waves specified in data

        :param data: dict with the following structure

                data[sample][state] = {'label1': line1,'label2': line2, ... ,'labelN',lineN}

    """

    samples     = list(data.keys())
    states      = list(data[samples[0]].keys())
    components  = list(data[samples[0]][states[0]])

    example = data[samples[0]][states[0]][components[0]][0]

    if np.asarray(example).ndim == 1:
        plot_type = '1D'
    elif np.asarray(example).ndim == 2:
        plot_type = '2D'
    else:
        print('>2D not supported')
        return

    fig, ax = plt.subplots(len(samples),len(states), figsize = (10, 16))

    for i, sample in enumerate(samples):

        for j, state in enumerate(states):

            for key, val in data[sample][state].items():

                if plot_type == '1D':

                    if type == 'librosa':
                        librosa.display.waveshow(y = val[0], sr = val[1], label=key, ax=ax[i,j],alpha = alpha, linewidth = linewidth)
                    else:
                        ax[i,j].plot(val[0], val[1],'o-',label=key,alpha = alpha, linewidth = linewidth)

                elif plot_type == '2D':

                    im = librosa.display.specshow( val[0], sr = val[1],
                                        x_axis= x_axis ,y_axis = y_axis, hop_length = hop_length, ax=ax[i,j], cmap = cmap)


            # Update figure (for later calculations)
            plt.tight_layout()

            # Column title
            ax[0,j].set_title(rf'${state}(z)$',fontsize=12)

            # y axis, label just at left and without offsetText (which is displayed in the axis label)
            plt.setp(ax[i,1].get_yticklabels(), visible=False)
            ax[i,j].yaxis.offsetText.set_visible(False)
            offset = ax[i,j].yaxis.get_major_formatter().get_offset()
            ax[i,0].set_ylabel(rf'{offset}'+'\n\n'+ ylabels[sample][state],labelpad=10,ha='right').set_rotation(0)
            ax[i,1].set_ylabel('') if plot_type == '2D' else False

            # x axis, ticks and labels on the bottom figure of each column

            plt.setp(ax[i,j].get_xticklabels(), visible=True, color='w')
            ax[i,j].set_xlabel('')

            plt.setp(ax[-1,j].get_xticklabels(), visible=True, color = 'k')
            ax[-1,j].set_xlabel('z [m]')

            # Add grid
            ax[i,j].grid()


    # Put a general legend at the bottom of the figure
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    # Make all y and x axis equal to conserve reference
    for i in range(len(samples)):
        for j in range(2):
            ax[0,0].get_shared_x_axes().join(ax[0,0], ax[i,j])
            ax[0,0].get_shared_y_axes().join(ax[0,0], ax[i,j])

    y_limits = [a.get_ylim() for a in  fig.axes[:-1]] ; y_limits = [item for tuple in y_limits for item in tuple]
    plt.setp(ax, ylim=(min(y_limits) , max(y_limits)))
    plt.figlegend(by_label.values(), by_label.keys(),loc='lower center',ncol=2,fancybox=False, shadow=False)

    if plot_type == '1D':
        # Subplots adjustment with no gaps
        plt.subplots_adjust(top=0.954,
                                bottom=0.091,
                                left=0.086,
                                right=0.985,
                                hspace=0.0,
                                wspace=0.0)

    if plot_type == '2D':
        # Add colorbar
        cbar_ax = fig.add_axes([0.9, 0.15, 0.025, 0.7])
        fig.colorbar(im, cax=cbar_ax, format="%+2.0f dB") if dB else fig.colorbar(im, cax=cbar_ax)

        # Subplots adjustment
        plt.subplots_adjust(top=0.954,
                            bottom=0.091,
                            left=0.091,
                            right=0.875,
                            hspace=0.095,
                            wspace=0.015)


def correlation2D(Z1,Z2,axis=1):

    """ Function to compute correlation along some axis between two 2D arrays

        : param Z1 (2D array): First 2D array to compute correlation
        : param Z2 (2D array): Second 2D array to compute correlation

        : optional axis (int): Axis to go

        : return corr2D (2D np.array): Array which contains correlation between arrays"""


    corr2D = list()
    for x in range(P.shape[axis]):

        Y1 = P[:,x]
        Y2 = S[:,x]

        # Normalization
        Y1 = (Y1 - np.mean(Y1)) / (np.std(Y1) * len(Y1))
        Y2 = (Y2 - np.mean(Y2)) / (np.std(Y2))

        # Cross corelation
        corr = np.correlate(Y1, Y2, mode='same')



    return np.array(corr2D)

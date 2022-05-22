import matplotlib.pyplot as plt
import matplotlib as mpl
import librosa
import librosa.display
from .Z9_utils import arr2librosa
import numpy as np


def Mel_spectrogram(samples,n_fft=2000, hop_length= 100,magnitude = 'module',cmap='jet'):

    print('\nSpectrogram:')
    print(' magnitude:', magnitude)
    print(' n_fft =',n_fft)
    print(' hop_length =',hop_length)

    fig, ax = plt.subplots(len(samples),2, figsize = (10, 16))

    for i, sample in zip(range(len(samples)),samples.values()):

        Temperature = sample.temperature
        Flecha = sample.flecha
        sr = int(1/(sample.z[1]-sample.z[0]))

        for j, state in enumerate(['P','S']):

            # np.array module to librosa stuff
            if magnitude == 'module' or magnitude == 'abs':
                wave, new_sr = arr2librosa(np.abs(sample.Data[j]),sr)
            elif magnitude == 'phase' or magnitude == 'angle':
                wave, new_sr = arr2librosa(np.angle(sample.Data[j]),sr)
            elif magnitude == 'real':
                wave, new_sr = arr2librosa(sample.Data[j].real,sr)
            elif magnitude == 'imag':
                wave, new_sr = arr2librosa(sample.Data[j].imag,sr)
            else:
                print('Invalid magnitude. Chose between: "module", "phase", "real" or "imag"')

            # Create the Mel Spectrograms
            wave = librosa.feature.melspectrogram(wave, sr=new_sr)

            # Amplitude to dB
            wave = librosa.amplitude_to_db(wave, ref = np.max)


            # Display real and imaginary values
            im = librosa.display.specshow( wave, sr = new_sr,
                    x_axis='time' ,y_axis = 'log', hop_length = hop_length, ax=ax[i,j], cmap = cmap)


            # Update figure (for later calculations)
            plt.tight_layout()

            # Column title
            ax[0,j].set_title(rf'${state}(z)$',fontsize=12)

            # y axis, label just at left and without offsetText (which is displayed in the axis label)
            plt.setp(ax[i,1].get_yticklabels(), visible=False)
            ax[i,j].yaxis.offsetText.set_visible(False)
            offset = ax[i,j].yaxis.get_major_formatter().get_offset()
            ylabel = rf'{offset}'+'\n\n'+ rf'$\nu \: [Hz]$'+'\n'+ rf'$T = {Temperature}\: Cº$'+'\n'+ rf'$\delta = {Flecha}\: mm$'
            ax[i,0].set_ylabel(ylabel,labelpad=10,ha='right').set_rotation(0)
            ax[i,1].set_ylabel('')


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


    cbar_ax = fig.add_axes([0.9, 0.15, 0.025, 0.7])
    fig.colorbar(im, cax=cbar_ax, format="%+2.0f dB")

    # Remove gap between subplots
    plt.subplots_adjust(top=0.954,
                        bottom=0.091,
                        left=0.091,
                        right=0.875,
                        hspace=0.095,
                        wspace=0.015)

    # Show the figure
    plt.show()

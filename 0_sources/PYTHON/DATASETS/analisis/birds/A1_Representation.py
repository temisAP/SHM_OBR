import matplotlib.pyplot as plt
import librosa
import librosa.display
from .Z9_utils import arr2librosa
import numpy as np


def Representation(samples):

    fig, ax = plt.subplots(len(samples),2, figsize = (10, 16))

    for i, sample in zip(range(len(samples)),samples.values()):

        Temperature = sample.temperature
        Flecha = sample.flecha
        sr = int(1/(sample.z[1]-sample.z[0]))

        for j, state in enumerate(['P','S']):

            # np.array to librosa stuff
            wave, new_sr = arr2librosa(sample.Data[j],sr)

            # Display real and imaginary values
            librosa.display.waveshow(y = wave.real, sr = new_sr, label='Real', ax=ax[i,j],  linewidth = 0.5)
            librosa.display.waveshow(y = wave.imag, sr = new_sr, label='Imaginary', ax=ax[i,j],  linewidth = 0.5)

            # Update figure (for later calculations)
            plt.tight_layout()

            # Column title
            ax[0,j].set_title(rf'${state}(z)$',fontsize=12)

            # y axis, label just at left and without offsetText (which is displayed in the axis label)
            plt.setp(ax[i,1].get_yticklabels(), visible=False)
            ax[i,j].yaxis.offsetText.set_visible(False)
            offset = ax[i,j].yaxis.get_major_formatter().get_offset()
            ylabel = rf'{offset}'+'\n\n'+ rf'$T = {Temperature}\: Cº$'+'\n'+ rf'$\delta = {Flecha}\: mm$'
            ax[i,0].set_ylabel(ylabel,labelpad=10,ha='right').set_rotation(0)


            # x axis, ticks and labels on the bottom figure of each column

            #true_range = [sample.z[0]*1e3, sample.z[-1]*1e3]
            #current_range = ax[i,j].get_xlim()
            def format_func(value,tick_idx):

                zfin    = float(true_range[1])
                z0      = float(true_range[0])
                xfin    = float(current_range[1])
                x0      = float(current_range[0])

                x = float(value)
                z = (x-x0) * (zfin-z0)/(xfin-x0)

                return round(z,3)

            #tick_labels = ax[i,j].get_xticks().tolist()
            #for idx, tick_label in enumerate(tick_labels):
            #    tick_labels[idx] = format_func(tick_label)
            #ax[i,j].set_xticklabels(tick_labels)


            #ax[i,j].xaxis.set_major_formatter(plt.FuncFormatter(format_func))
            plt.setp(ax[i,j].get_xticklabels(), visible=True, color='w')
            ax[i,j].set_xlabel('')
            #ax[i,j].xaxis.set_major_locator(plt.LinearLocator(10))
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

    # Remove gap between subplots
    plt.subplots_adjust(top=0.954,
                            bottom=0.091,
                            left=0.086,
                            right=0.985,
                            hspace=0.0,
                            wspace=0.0)

    # Show the f*ckin figure, yeah!
    plt.show()

import librosa
from scipy.io.wavfile import write
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def a_plot(data,ylabels,
            type='librosa', alpha = 1,  linewidth = 0.5,linestyle = '-o',
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
                        ax[i,j].plot(val[0], val[1],linestyle,label=key,alpha = alpha, linewidth = linewidth)

                elif plot_type == '2D':

                    im = librosa.display.specshow( val[0], sr = val[1],
                                        x_axis= x_axis ,y_axis = y_axis, hop_length = hop_length, ax=ax[i,j], cmap = cmap)


            # Update figure (for later calculations)
            plt.tight_layout()

            # Column title
            ax[0,j].set_title(state,fontsize=12)

            # y axis, label just at left and without offsetText (which is displayed in the axis label)


            plt.setp(ax[i,j].get_yticklabels(), visible=False) if j!=0 else None
            ax[i,j].set_ylabel('') if plot_type == '2D' else False

            ax[i,j].yaxis.offsetText.set_visible(False)
            offset = ax[i,j].yaxis.get_major_formatter().get_offset()

            ax[i,0].set_ylabel(rf'{offset}'+'\n\n'+ ylabels[sample][state],labelpad=10,ha='right').set_rotation(0)

            # x axis, ticks and labels on the bottom figure of each column

            plt.setp(ax[i,j].get_xticklabels(), visible=True, color='w')
            ax[i,j].set_xlabel('')

            plt.setp(ax[-1,j].get_xticklabels(), visible=True, color = 'k')
            ax[-1,j].set_xlabel('z [m]')

            # Add grid
            ax[i,j].grid() if plot_type == '1D' else None




    # Make all y and x axis equal to conserve reference
    for i in range(len(samples)):
        for j in range(len(states)):
            ax[0,0].get_shared_x_axes().join(ax[0,0], ax[i,j])
            ax[0,0].get_shared_y_axes().join(ax[0,0], ax[i,j])

    y_limits = [a.get_ylim() for a in  fig.axes[:-1]] ; y_limits = [item for tuple in y_limits for item in tuple]
    plt.setp(ax, ylim=(min(y_limits) , max(y_limits)))

    # Put a general legend at the bottom of the figure
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.figlegend(by_label.values(), by_label.keys(),loc='lower center',ncol=2,fancybox=False, shadow=False) if plot_type == '1D' else None

    if plot_type == '1D':
        # Subplots adjustment with no gaps
        plt.subplots_adjust(top=0.954,
                                bottom=0.091,
                                left=0.096,
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
                            left=0.096 if len(states)==2 else 0.096,
                            right=0.875,
                            hspace=0.095,
                            wspace=0.015)

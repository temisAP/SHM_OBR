import librosa
from scipy.io.wavfile import write
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def b_plot(data,ylabels,
                cmap='jet',x_axis = 'time', y_axis = 'log', hop_length=None):

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

    fig, ax = plt.subplots(len(samples),2, figsize = (10, 16))

    for i, sample in enumerate(samples):

        j = 0

        if j==0:
            P, sr = data[sample][states[0]][components[0]]
            S, sr = data[sample][states[1]][components[0]]

            corr = correlation2D(P,S,axis=1)

            im = librosa.display.specshow(corr, sr = sr,
                                    x_axis= x_axis ,y_axis = y_axis, hop_length = hop_length, ax=ax[i,j], cmap = cmap)

            ax[0,j].set_title(r'$P \star S$',fontsize=12)

        j = 1
        if j == 1:
            if i == 0:
                base_corr = corr

            d_corr = corr-base_corr

            im = librosa.display.specshow(d_corr, sr = sr,
                                    x_axis= x_axis ,y_axis = y_axis, hop_length = hop_length, ax=ax[i,j], cmap = cmap)

            ax[0,j].set_title(r'$(P \star S) - (P_0 \star S_0)$',fontsize=12)


        # Update figure (for later calculations)
        plt.tight_layout()

        for j in range(2):

            # y axis, label just at left and without offsetText (which is displayed in the axis label)

            plt.setp(ax[i,j].get_yticklabels(), visible=False) if j!=0 else None
            ax[i,j].set_ylabel('') if plot_type == '2D' else False

            ax[i,j].yaxis.offsetText.set_visible(False)
            offset = ax[i,j].yaxis.get_major_formatter().get_offset()

            ax[i,0].set_ylabel(rf'{offset}'+'\n\n'+ ylabels[sample][states[0]],labelpad=10,ha='right').set_rotation(0)

            # x axis, ticks and labels on the bottom figure of each column

            plt.setp(ax[i,j].get_xticklabels(), visible=True, color='w')
            ax[i,j].set_xlabel('')

            plt.setp(ax[-1,j].get_xticklabels(), visible=True, color = 'k')
            ax[-1,j].set_xlabel('z [m]')





    # Make all y and x axis equal to conserve reference
    for i in range(len(samples)):
        for j in range(2):
            ax[0,0].get_shared_x_axes().join(ax[0,0], ax[i,j])
            ax[0,0].get_shared_y_axes().join(ax[0,0], ax[i,j])

    y_limits = [a.get_ylim() for a in  fig.axes[:-1]] ; y_limits = [item for tuple in y_limits for item in tuple]
    plt.setp(ax, ylim=(min(y_limits) , max(y_limits)))

    # Put a general legend at the bottom of the figure
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.figlegend(by_label.values(), by_label.keys(),loc='lower center',ncol=2,fancybox=False, shadow=False) if plot_type == '1D' else None

    # Add colorbar
    cbar_ax = fig.add_axes([0.9, 0.15, 0.025, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    # Subplots adjustment
    plt.subplots_adjust(top=0.954,
                        bottom=0.091,
                        left=0.091,
                        right=0.875,
                        hspace=0.095,
                        wspace=0.015)

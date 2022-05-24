import matplotlib.pyplot as plt
import matplotlib as mpl
import librosa
import librosa.display
from .Z9_utils import arr2librosa
import numpy as np
import os
import pandas as pd

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from UTILS.utils import printProgressBar


def Zero_crossing_rate(self,sample_size = None):


    slices = self.slices

    conditions_file     = os.path.join(self.path,self.folders['4_INFORMATION'],self.INFO['conditions filename'])
    self.conditions_checkout()
    conditions_df       = pd.read_csv(conditions_file)
    T0      = self.obrfiles[conditions_df['default state'][0]].temperature
    delta0  = self.obrfiles[conditions_df['default state'][0]].flecha
    L = float(conditions_df['L\n[mm]'][0])  * 1e-3               # mm to m
    t = float(conditions_df['t\n[mm]'][0])  * 1e-3               # mm to m
    alpha = float(conditions_df['alpha\n[µm/(m·K)]'])            # already microdeformations

    if sample_size:
        from random import sample
        keys = sample(slices.keys(),sample_size)
        new_slices = dict.fromkeys(keys)
        for key in keys:
            new_slices[key] = slices[key]
        slices = new_slices

    total_len = len(slices.keys()) * 4; current_len = 1

    fig, ax = plt.subplots(2,2, figsize = (10, 16))

    for j, state in enumerate(['P','S']):


        for i, c_axis in enumerate(['Real','Imaginary']):

            delta_T_list = list()
            delta_EPS_list = list()
            zero_list = list()

            for key, slice in slices.items():

                data = getattr(slice, state)
                if c_axis == 'Real':
                    wave, new_sr = arr2librosa(data.real,len(data.real))
                elif c_axis == 'Imaginary':
                    wave, new_sr = arr2librosa(data.imag,len(data.imag))

                # Access status information of the slice to create outputs for NN

                delta_T         = slice.temperature - T0    # K
                delta_flecha    = (slice.flecha - delta0) * 1e-3      # mm to m
                x               = slice.x * 1e-3                               # mm to m

                eps_mec = 3*delta_flecha*t/(2*L**3) * (x-L) * 1e6               # Mechanical microdeformations
                eps_the = alpha * delta_T                                       # Thermal  microdeformations
                delta_EPS = eps_mec + eps_the                                   # Total microdeformations

                # Zero crossings
                zero_wave = librosa.zero_crossings(y=wave, pad=False)

                # Assignation to later plot

                delta_T_list.append(delta_T)
                delta_EPS_list.append(delta_EPS)
                zero_list.append(sum(zero_wave))


            # Display real and imaginary values
            im = ax[i,j].scatter(delta_T_list,delta_EPS_list,c=zero_list,marker = 'x',cmap='coolwarm')

            # Update figure (for later calculations)
            plt.tight_layout()

            # Title
            ax[i,j].set_title(rf'${state},{c_axis}$',fontsize=12)

            # y axis, label just at left and without offsetText (which is displayed in the axis label)
            plt.setp(ax[i,1].get_yticklabels(), visible=False)
            ax[i,j].yaxis.offsetText.set_visible(False)
            offset = ax[i,j].yaxis.get_major_formatter().get_offset()
            ylabel = rf'$\Delta \mu \varepsilon$'
            ax[i,0].set_ylabel(ylabel,labelpad=10,ha='right').set_rotation(0)
            ax[i,1].set_ylabel('')


            # x axis, ticks and labels on the bottom figure of each column

            plt.setp(ax[i,j].get_xticklabels(), visible=True, color='k')
            ax[i,j].set_xlabel( rf'$\Delta T \:$ [K]')

            # Print progress bar
            printProgressBar(current_len, total_len); current_len += 1


    # Put a general legend at the bottom of the figure
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    # Make all y and x axis equal to conserve reference
    for i in range(2):
        for j in range(2):
            ax[0,0].get_shared_x_axes().join(ax[0,0], ax[i,j])
            ax[0,0].get_shared_y_axes().join(ax[0,0], ax[i,j])
            # Add grid
            ax[i,j].grid()

    y_limits = [a.get_ylim() for a in  fig.axes[:-1]] ; y_limits = [item for tuple in y_limits for item in tuple]
    plt.setp(ax, ylim=(min(y_limits) , max(y_limits)))
    plt.figlegend(by_label.values(), by_label.keys(),loc='lower center',ncol=2,fancybox=False, shadow=False)

    # Add colorbar
    cbar_ax = fig.add_axes([0.9, 0.15, 0.025, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    # Remove gap between subplots
    plt.subplots_adjust(top=0.954,
                        bottom=0.061,
                        left=0.086,
                        right=0.885,
                        hspace=0.175,
                        wspace=0.0)

    # Show the figure
    plt.show()

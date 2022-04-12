import os
import pandas as pd
import numpy as np
import glob
import re
import sys
import time
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from UTILS.utils import find_index
from UTILS.read_obr import multi_read_obr
from SIGNAL.Spectral_Shift import global_spectral_shift
from SIGNAL.wavelet import wavelet
from SIGNAL.chirplet import chirplet
from SIGNAL.STFT import stft
from SIGNAL.spectrogram import spectrogram


def analysis_0(self,files,position):

    # Compute chirplets

    REF = files[0]
    chirplets = dict.fromkeys(files)
    for file in files:
        y = self.obrfiles[file].Data[0]
        f = self.obrfiles[file].f
        i = find_index(self.obrfiles[file].z,position)

        window = 1000

        y = y[i-window:i+window]

        chirplets[file] = chirplet(y,f,plot=False)
        #wavelet(y,plot=True)

        plt.title(file)
        plt.close()

    plt.show()

def analysis_1(self,files,position):

    # Compute chirplets

    REF = files[0]
    chirplets = dict.fromkeys(files)
    for file in files:
        y = self.obrfiles[file].Data[0]
        i = find_index(self.obrfiles[file].z,position)

        window = 1000

        y = y[i-window:i+window]

        chirplets[file] = chirplet(np.abs(y),plot=False)
        #wavelet(y,plot=True)

        plt.title(file)
        plt.close()



    # 2d Correlation of chirplets

    for file in files:


        corr = signal.correlate2d(chirplets[REF], chirplets[file], boundary='wrap', mode='full') # mode full es clave (la diferencia es más marcada) , fill bounds hace que sea redondo, wrap hace bandas y symmetry hace que se pierda la simetría xd
        y, x = np.unravel_index(np.argmax(corr), corr.shape)  # find the match

        import matplotlib.pyplot as plt

        v_max = 6e-5 # np.amax(corr)
        v_min = 1e-5 # np.amin(corr)

        fig, (ax_orig, ax_template, ax_corr) = plt.subplots(3, 1)

        ax_orig.contour(chirplets[REF], cmap='jet')
        sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=np.amin(chirplets[REF]), vmax=np.amax(chirplets[REF])))
        cbar = plt.colorbar(sm,ax=ax_orig,spacing='proportional')
        ax_orig.set_title(REF)
        ax_orig.grid()

        ax_template.contour(chirplets[file], cmap='jet')
        sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=np.amin(chirplets[file]), vmax=np.amax(chirplets[file])))
        cbar = plt.colorbar(sm,ax=ax_template,spacing='proportional')
        ax_template.set_title(file)
        ax_template.grid()


        ax_corr.contour(corr, cmap='jet')
        sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=v_min, vmax=v_max))
        cbar = plt.colorbar(sm,ax=ax_corr,spacing='proportional')
        ax_corr.set_title('Cross-correlation')
        ax_corr.plot(x, y, 'ko')
        ax_corr.grid()

        print(x,y)

        fig.tight_layout()


    plt.show()


    if False:
        import csv
        with open(os.path.join(self.path,self.folders['1_PROCESSED'],'data1.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(y.real)
            writer.writerow(y.imag)

def analysis_2(self,files,position):

    import matplotlib.pyplot as plt
    import scipy.signal as signal

    v_max = 6e-5
    v_min = 1e-5

    delta = 0.05
    positions = [position-delta,position,position+delta]
    data = {f'{positions[0]}':dict.fromkeys(files),
            f'{positions[1]}':dict.fromkeys(files),
            f'{positions[2]}':dict.fromkeys(files)}

    for pos in positions:

        # Compute chirplets
        for file in files:
            y = self.obrfiles[file].Data[0]
            i = find_index(self.obrfiles[file].z,pos)
            window = 1000
            data[str(pos)][file] = chirplet(np.abs(np.fft.fft(y[i-window:i+window])),plot=False)

    # 2d Correlation of chirplets
    REF = files[0]
    fig, ax = plt.subplots(3, 4)

    idx = 0
    for key in data.keys():
        jdx = 0
        for file in files:

            corr = signal.correlate2d(data[key][REF], data[key][file], boundary='symm', mode='same') # mode full es clave (la diferencia es más marcada) , fill bounds hace que sea redondo, wrap hace bandas y symmetry hace que se pierda la simetría xd
            y, x = np.unravel_index(np.argmax(corr), corr.shape)  # find the match

            ax[idx%3,jdx%4].contour(corr, cmap='jet')
            sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=v_min, vmax=v_max))
            cbar = plt.colorbar(sm,ax=ax[idx%3,jdx%4],spacing='proportional')
            ax[idx%3,jdx%4].set_title(f'{file} at {key} m')
            ax[idx%3,jdx%4].plot(x, y, 'ko')
            ax[idx%3,jdx%4].grid()

            print(f'{file} \t at {key}: {x} {y}')

            fig.tight_layout()
            jdx += 1
        idx += 1


    plt.show()

def analysis_3(self,files,position):

    import matplotlib.pyplot as plt
    import scipy.signal as signal

    v_max = 6e-5
    v_min = 1e-5

    delta = 0.05
    positions = [position-delta,position,position+delta]
    data = {f'{positions[0]}':dict.fromkeys(files),
            f'{positions[1]}':dict.fromkeys(files),
            f'{positions[2]}':dict.fromkeys(files)}

    for pos in positions:

        # Compute chirplets
        for file in files:
            y = self.obrfiles[file].Data[0]
            i = find_index(self.obrfiles[file].z,pos)
            window = 1000
            data[str(pos)][file] = chirplet(np.abs(np.fft.fft(y[i-window:i+window])),plot=False)

    # 2d Correlation of chirplets
    REF = files[0]
    fig, ax = plt.subplots(3, 4)

    idx = 0
    for key in data.keys():
        jdx = 0
        for file in files:

            ax[idx%3,jdx%4].contour(data[key][file], cmap='jet')
            sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=v_min, vmax=v_max))
            cbar = plt.colorbar(sm,ax=ax[idx%3,jdx%4],spacing='proportional')
            ax[idx%3,jdx%4].set_title(f'{file} at {key} m')
            ax[idx%3,jdx%4].grid()


            fig.tight_layout()
            jdx += 1
        idx += 1


    plt.show()

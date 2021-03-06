import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from UTILS.utils import printProgressBar


""" Spectral shift

    This module contains functions used to compute the relaive spectralshift
    (which actually is spectralshift/central_frecuency) along an optic fiber longitude

"""

def local_spectral_shift(y1,y2,f,display=False,fft=True):
    """ Function to compute the relative spectral shift in a region.

        Scipy signal.correlate is used to determine the cross correlation, then the
        maximum position is located and related with the spectral shift trought the scan ratio

        :param  y1 (np.array)   : First signal to compare, used as reference
        :param  y2 (np.array)   : Second signal to compare
        :param  f  (np.array)   : Frequency domain x axis [GHz]

        :retruns spectralshift/mean_f (float)   : relative spectralshift
    """

    type = '1D' if len(y1.shape) == 1 else 'ND'

    n_points = len(y1) if type == '1D' else y1.shape[1]


    # Frequency sampling
    DF = f[-1]-f[0]     # Frequency increment
    n = n_points        # Sample lenght
    sr = 1/(DF/n)       # Scan ratio

    # FFT
    if type == '1D':
        Y1 = np.absolute(np.fft.fft(y1)) if fft else y1
        Y2 = np.absolute(np.fft.fft(y2)) if fft else y2
        Y1 = (Y1 - np.mean(Y1)) / (np.std(Y1) * len(Y1))
        Y2 = (Y2 - np.mean(Y2)) / (np.std(Y2))
    else:
        Y1 = np.absolute(np.fft.fftn(y1))
        Y2 = np.absolute(np.fft.fftn(y2))
        print('Under construction')
        exit()
        #Y1 = (Y1 - np.mean(Y1)) / (np.std(Y1) * len(Y1))
        #Y2 = (Y2 - np.mean(Y2)) / (np.std(Y2))

    # Cross corelation
    corr = np.correlate(Y1, Y2, mode='same') if type == '1D' else signal.correlate(Y1, Y2, mode='same')

    # Spectral shift
    spectralshift_arr = np.linspace(-0.5*n/sr, 0.5*n/sr, n+1)
    spectralshift = spectralshift_arr[np.argmax(corr)]

    if display:

        dz = 2e-6
        df = DF
        z_window = np.linspace(-len(y1)/2*dz,len(y1)/2*dz,len(y1))
        f_window = np.linspace(f[0],f[-1],len(y1))

        plt.figure()
        plt.title('Time domain')
        plt.plot(z_window,y1)
        plt.plot(z_window,y2)
        plt.xlabel(r'$\Delta$ z [mm]')
        plt.legend(['Reference signal','Signal after deformation/temperature increment'])
        plt.grid()

        plt.figure()
        plt.title('Frequency domain')
        plt.plot(f_window,Y1)
        plt.plot(f_window,Y2)
        plt.xlabel(r'$\nu$ [GHz]')
        plt.legend(['Reference signal','Signal after deformation/temperature increment'])
        plt.grid()

        plt.figure()
        plt.title('Cross correlation')
        plt.plot(spectralshift_arr,np.interp(spectralshift_arr,np.linspace(-len(corr)/2,len(corr)/2,len(corr)),corr))
        plt.xlabel(r'$\Delta\nu$ [GHz]')
        plt.grid()

        plt.show()


    return -1*spectralshift/np.mean(f)


def global_spectral_shift(y1,y2,f,delta=200,window=1000,display = False,progressbar=False,fft=True):
    """ Computes relative spectral shift (spectralshift) between two signals in a given window
        :param y1       (np.array)  : first signal
        :param y2       (np.array)  : second signal
        :param delta    (int)       : index step (optional)
        :param window   (int)       : index window (optional)

        :returns spectralshift (np.array)   : array with relative spectral shifts
    """

    type = '1D' if len(y1.shape) == 1 else 'ND'

    n_points = len(y1) if type == '1D' else y1.shape[1]

    spectralshift = []
    steps = range(window,n_points-window+1,delta)

    if display == True:

        plt.figure()
        plt.plot(y1)
        plt.plot(y2)
        plt.grid()
        plt.show()

        point = float(input('Point to analize:'))
        point = min(steps, key=lambda x:abs(x-point))


    for i in steps:

            if type == '1D':
                yy1 = y1[i-window:i+window]
                yy2 = y2[i-window:i+window]

            else:
                yy1 = y1[:,i-window:i+window]
                yy2 = y2[:,i-window:i+window]

            diff = local_spectral_shift(yy1,yy2,f,fft=fft)
            spectralshift.append(float(diff))

            if display==True and i==point:
                diff = local_spectral_shift(yy1,yy2,f,fft=fft,display=True)


            printProgressBar(i + 1, len(y1)-window-delta-1, prefix = 'Computing spectral shift:', suffix = 'Complete', length = 50) if progressbar else None


    spectralshift = np.array(spectralshift)

    return spectralshift

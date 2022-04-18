import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import signal
import sys
import os

# For model loading
import pickle
import torch
try:
    import keras
    from keras.models import load_model
    from keras.models import model_from_json
except ImportError as e:
    print('Warning: Failed to import keras. Ignore warning message if the model is not keras based')

# Custom modules
from .utils import printProgressBar
sys.path.append(os.path.join(os.path.dirname(__file__), '../0_sources/PYTHON'))
from IA import IA


def local_sensor(y1,y2,f,IA_obj,display=False):
    """ Function to predict temperature and deformation in a narrow region.

        Scipy signal.correlate is used to determine the cross correlation and autocorrelations, then the
        maximum position is located and related with the spectral shift trought the scan ratio.
        Before a deep learning model will predict the temperature and deformation.

        :param  y1 (np.array)       : First signal to compare, used as reference
        :param  y2 (np.array)       : Second signal to compare
        :param  f  (np.array)       : Frequency domain x axis [GHz]
        :param  IA_obj (IA object)  : IA object which contains all the information for temperature and deformation prediction

        :returns T (float)   : Temperature increment    [K]
        :returns E (float)   : Microdeformation increment
    """

    mode = 'same'

    """ Autocorrelation """

    # Autocorrelation
    y = y1
    autocorr1 = np.correlate(y, y, mode=mode)/np.var(y)/len(y)
    y = y2
    autocorr2 = np.correlate(y, y, mode=mode)/np.var(y)/len(y)
    autocorr = np.absolute(autocorr1-autocorr2)

    # STL filter
    seasonal,trend = np.array(sm.tsa.filters.hpfilter(autocorr, lamb=25))
    autocorr = trend

    autocorr = autocorr[int(len(autocorr)/2-200):int(len(autocorr)/2+200)]

    """ Cross correlation """

    # Frequency sampling
    DF = f[-1]-f[0]     # Frequency increment
    n = len(y1)         # Sample lenght
    sr = 1/(DF/n)       # Scan ratio

    # FFT
    Y1 = np.absolute(np.fft.fft(y1))
    Y2 = np.absolute(np.fft.fft(y2))
    Y1 = (Y1 - np.mean(Y1)) / (np.std(Y1) * len(Y1))
    Y2 = (Y2 - np.mean(Y2)) / (np.std(Y2))

    # Cross corelation
    crosscorr = np.correlate(Y1, Y2, mode=mode)

    # Spectral shift
    spectralshift_lags = np.linspace(-0.5*n/sr, 0.5*n/sr, n+1)
    spectralshift = spectralshift_lags[np.argmax(crosscorr)]
    spectralshift = -1*spectralshift/np.mean(f)*1e6             # micro spectralshift

    """ Model """

    # Sizes
    crosscorr = np.interp(np.linspace(0,1,len(crosscorr)),np.linspace(0,1,2000), crosscorr)
    autocorr  = np.interp(np.linspace(0,1,len(autocorr)) ,np.linspace(0,1,400),  autocorr )

    # Scalers
    spectralshift   = IA_obj.scaler_ss.transform(spectralshift)
    croscorr        = IA_obj.scaler_cc.transform(crosscorr)
    autocorr        = IA_obj.scaler_ac.transform(autocorr)

    t = [[spectralshift],crosscorr,autocorr]
    x = np.array([item for sublist in t for item in sublist])

    # Prediction
    T = float(IA_obj.scaler_T.inverse_transform(IA_obj.model_T.predict(x.reshape(1, -1))))
    E = float(IA_obj.scaler_E.inverse_transform(IA_obj.model_E.predict(x.reshape(1, -1))))

    if display:
        print('Under construction')

    return T,E


def sensor(y1,y2,f,delta=200,window=1000,display = False):
    """ Computes relative spectral shift (spectralshift) between two signals in a given window
        :param y1       (np.array)  : first signal
        :param y2       (np.array)  : second signal
        :param delta    (int)       : index step (optional)
        :param window   (int)       : index window (optional)

        :returns spectralshift (np.array)   : array with relative spectral shifts
    """

    print('*Loading IA model')
    IA_obj = IA('.',name='./IA')

    Ts = list()
    Es = list()
    steps = range(window,len(y1)-window+1,delta)

    if display == True:

        plt.figure()
        plt.plot(y1)
        plt.plot(y2)
        plt.grid()
        plt.show()

        point = float(input('Point to analize:'))
        point = min(steps, key=lambda x:abs(x-point))


    printProgressBar(0, len(y1)-window-delta-1, prefix = 'Progress:', suffix = 'Complete', length = 50)
    for i in steps:

            yy1 = y1[i-window:i+window]
            yy2 = y2[i-window:i+window]

            T,E = local_sensor(yy1,yy2,f,IA_obj)
            Ts.append(float(T))
            Es.append(float(E))

            if display==True and i==point:
                diff = local_sensor(yy1,yy2,f,IA_obj,display=True)

            printProgressBar(i + 1, len(y1)-window-delta-1, prefix = 'Progress:', suffix = 'Complete', length = 50)

    print('*Relative spectral shift done!')

    T = np.array(Ts)
    E = np.array(Es)

    return T,E

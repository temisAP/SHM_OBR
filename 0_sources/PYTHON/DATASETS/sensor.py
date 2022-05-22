import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import signal
import sys
import os
import time
import torch

# Custom modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from UTILS.utils import printProgressBar


def local_sensor(self,P1,S1,P2,S2,f,display=False):
    """ Function to predict temperature and deformation in a narrow region.

        Scipy signal.correlate is used to determine the cross correlation and autocorrelations, then the
        maximum position is located and related with the spectral shift trought the scan ratio.
        Before a deep learning model will predict the temperature and deformation.

        :param  P1 (np.array)       : Current state signal (p-polarization)
        :param  S1 (np.array)       : Current state signal (s-polarization)
        :param  P2 (np.array)       : Reference state signal (p-polarization)
        :param  S2 (np.array)       : Reference state signal (s-polarization)
        :param  f  (np.array)       : Frequency domain x axis [GHz]

        :returns T (float)   : Temperature increment        [K]
        :returns E (float)   : Microdeformation increment   [με]
    """

    """ Frequency sampling """
    DF = f[-1]-f[0]     # Frequency increment
    n = len(P1)         # Sample lenght
    sr = 1/(DF/n)       # Scan ratio
    spectralshift_arr = np.linspace(-0.5*n/sr, 0.5*n/sr, n+1)

    secuence1 = [P1,P1,P2,S1,S1,S2]
    secuence2 = [P1,P2,P2,S1,S2,S2]

    Df = 1/sr

    t = list()

    """ Cross correlations """
    for y1,y2 in zip(secuence1,secuence2):

        # FFT
        Y1 = np.absolute(np.fft.fft(y1))
        Y2 = np.absolute(np.fft.fft(y2))
        Y1 = (Y1 - np.mean(Y1)) / (np.std(Y1) * len(Y1))
        Y2 = (Y2 - np.mean(Y2)) / (np.std(Y2))

        # Cross corelation
        t.extend(np.correlate(Y1, Y2, mode='same').tolist())

    """ Input """
    t = [t,[Df]] # Append frequency step to have information about magnitude
    X = np.array([item for sublist in t for item in sublist])
    X = self.IA_obj.scalerX.transform(X)

    """ Call model """

    try:
        X = torch.from_numpy( np.array(X.reshape(1,-1)) ).float()
        T,E = self.IA_obj.model(X)
    except:
        X = torch.from_numpy( np.array(X.reshape(1,-1)) ).float().to(self.IA_obj.device) # Torch tensor in the same device as the model
        T,E = self.IA_obj.model(X)

    predictions = self.IA_obj.scalerY.inverse_transform([T,E])

    T = float( predictions[0][0].cpu().detach().numpy() ) + 0.6
    E = float( predictions[0][1].cpu().detach().numpy() ) - 27.0

    if display:
        print('Plot under construction')

    return T,E


def sensor(self,Data,refData,f,delta=200,window=1000,display = False):

    """ Computes relative spectral shift (spectralshift) between two signals in a given window
        :param Data     (np.array)  : current state signals
        :param refData  (np.array)  : reference state signals
        :param delta    (int)       : index step (optional)
        :param window   (int)       : index window (optional)

        :returns spectralshift (np.array)   : array with relative spectral shifts
    """

    Ts = list()
    Es = list()

    window = 1000
    steps = range(window,len(Data[0])-window+1,delta)

    if display == True:

        plt.figure()
        plt.plot(Data[2])
        plt.plot(refData[2])
        plt.grid()
        plt.show()

        point = float(input('Point to analize:'))
        point = min(steps, key=lambda x:abs(x-point))

    zero_time = float(time.time());measure = True

    printProgressBar(0, len(Data[0])-window-delta-1, prefix = 'Progress:', suffix = 'Complete', length = 50)
    for i in steps:

            P1 = Data[0][i-window:i+window]
            S1 = Data[1][i-window:i+window]
            P2 = refData[0][i-window:i+window]
            S2 = refData[1][i-window:i+window]

            T,E = self.local_sensor(P1,S1,P2,S2,f,
                                    display=True if (display==True and i==point) else False)

            Ts.append(float(T))
            Es.append(float(E))

            printProgressBar(i + 1, len(Data[0])-window-delta-1, prefix = 'Progress:', suffix = 'Complete', length = 50)


    print('*Segment computed!')

    seasonal,T = np.array(sm.tsa.filters.hpfilter(np.array(Ts), lamb=199))
    seasonal,E = np.array(sm.tsa.filters.hpfilter(np.array(Es), lamb=199))

    return T,E

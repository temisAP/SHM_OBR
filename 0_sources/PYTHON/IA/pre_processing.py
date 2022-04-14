import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

class a_scaler(object):
    def __init__(self,max,min):
        self.max = max
        self.min = min
        if self.max == self.min:
            print('Scaler is broken :(')
            self.max = 1
            self.min = 0

    def transform(self,x):
        return (x-self.min)/(self.max-self.min)

    def inverse_transform(self,z):
        return (self.max-self.min) * z + self.min


def pre_processing(self,X,y,plot_preprocessing=False,plot_histogram=False):

    # Lists to np.arrays

    for key,val in X.items():
        X[key] = np.array(X[key])
        y[key] = np.array(y[key])

    # Set up scalers

    xx = y['train'][:].flatten().reshape(-1, 1)
    self.scaler_Y  = scaler_Y  = a_scaler(xx.max(),xx.min())


    # Transform each sample

    for key,val in y.items():

        # Transform each sample
        yy = np.array([scaler_Y.transform(sample)  for sample in y[key][:]])

        # Plot histograms of values of ss, cc, ac, T and E
        if plot_histogram:

            fig, axs = plt.subplots(1,1, constrained_layout=True)

            axs.hist(yy,bins=20)
            axs.set_xlim(-0.5,1.5)
            axs.grid()
            axs.set_title('SS')

            fig.suptitle(f'Histogram for {key}')

        plt.show()

        # Concatenate information
        y[key] =  np.transpose(np.array([yy]))

    return X,y

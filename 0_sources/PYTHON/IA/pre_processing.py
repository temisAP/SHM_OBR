import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

class a_scaler(object):
    def __init__(self,X):
        print('Creating scaler for',X.shape[1],'items and',X.shape[0],'samples')
        self.max = np.zeros(X.shape[1])
        self.min = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            self.max[i] = np.amax(X[:,i])
            self.min[i] = np.amin(X[:,i])
            if self.max[i] == self.min[i]:
                print('Column',i,'with single value')
                self.max[i] = np.amax(X[:,i])*2
                self.min[i] = 0

    def transform(self,x):
        out = np.empty_like(x)
        for i in range(x.shape[1]):
            out[:,i] = (x[:,i]-self.min[i])/(self.max[i]-self.min[i])
        return out

    def inverse_transform(self,z):
        out = np.empty_like(z)
        for i in range(z.shape[1]):
            out[:,i] = (self.max[i]-self.min[i]) * z[:,i] + self.min[i]

        return out


def pre_processing(self,plot_preprocessing=False,plot_histogram=False):

    # Lists to np.arrays

    for key,val in self.X.items():
        self.X[key] = np.array(self.X[key])
        self.Y[key] = np.array(self.Y[key])
        print(f'{key} size = {len(self.X[key])}')

    # Set up scalers
    self.scalerX = scalerX = a_scaler(self.X['train'])
    self.scalerY = scalerY = a_scaler(self.Y['train'])

    # Transform each sample
    for key,val in self.X.items():

        self.X[key] = scalerX.transform(self.X[key])
        self.Y[key] = scalerY.transform(self.Y[key])


        if plot_histogram:

            fig, axs = plt.subplots(3,2, constrained_layout=True)

            limits = np.array([0,2,4,6,8,10,12]) * 1000; idx = 0

            for i in range(6):

                for k in range(len(limits)-1):
                    axs[i//2,i%2].hist(self.X[key][:,limits[k]:limits[k+1]],bins=20,histtype=u'step')
                axs[i//2,i%2].set_xlim(-0.5,1.5)
                axs[i//2,i%2].set_xlim(-0.5,1.5)
                axs[i//2,i%2].grid()
                axs[i//2,i%2].set_title(idx)
                idx += 1

            fig.suptitle(f'Histograms for {key}: inputs')

            fig, axs = plt.subplots(1, 2, constrained_layout=True)

            axs[0].hist(self.Y[key][:,0],bins=20)
            axs[0].set_xlim(-0.5,1.5)
            axs[0].grid()
            axs[0].set_title('T')

            axs[1].hist(self.Y[key][:,1],bins=20)
            axs[1].set_xlim(-0.5,1.5)
            axs[1].grid()
            axs[1].set_title('E')

            fig.suptitle(f'Histograms for {key}: outputs')

        plt.show()

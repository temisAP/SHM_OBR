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

    xx = X['train'][-1:].flatten().reshape(-1,1)
    self.scaler_F = scaler_F = a_scaler(xx.max(),xx.min())

    xx = y['train'][:,0].flatten().reshape(-1, 1)
    self.scaler_T  = scaler_T  = a_scaler(xx.max(),xx.min())
    xx = y['train'][:,1].flatten().reshape(-1, 1)
    self.scaler_E  = scaler_E  = a_scaler(xx.max(),xx.min())

    # Transform each sample

    for key,val in X.items():

        # Transform each sample
        freq   = np.array([ scaler_F.transform(float(sample)) for sample in X[key][:,-1:]])

        T      = np.array([ scaler_T.transform(float(sample)) for sample in y[key][:,0]])
        E      = np.array([ scaler_E.transform(float(sample)) for sample in y[key][:,1]])

        X[key] = np.concatenate((X[key][:,:-1],np.array(np.transpose([freq]))),axis=1)
        y[key] = np.concatenate((np.transpose(np.array([T])),np.transpose(np.array([E]))),axis=1)


        # Plot histograms of values of x, T and E
        if plot_histogram:

            fig, axs = plt.subplots(2, 2, constrained_layout=True)

            axs[0,0].hist(X[key][:,:-1].flatten(),bins=20)
            axs[0,0].set_xlim(-0.2,1.2)
            axs[0,0].grid()
            axs[0,0].set_title('x')

            axs[0,1].hist(X[key][:,-1:],bins=20)
            axs[0,1].set_xlim(-0.2,1.2)
            axs[0,1].grid()
            axs[0,1].set_title('freq')

            axs[1,0].hist(y[key][:,0],bins=20)
            axs[1,0].set_xlim(-0.2,1.2)
            axs[1,0].grid()
            axs[1,0].set_title('T')

            axs[1,1].hist(y[key][:,1],bins=20)
            axs[1,1].set_xlim(-0.2,1.2)
            axs[1,1].grid()
            axs[1,1].set_title('E')

            fig.suptitle(f'Histograms for {key}')

        plt.show()


    return X,y

# **IMPORTS**"""

# Commented out IPython magic to ensure Python compatibility.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from __future__ import print_function, division
torch.manual_seed(0)
# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

import matplotlib.pyplot as plt
import pandas as pd
from skimage import io, transform
import numpy as np
from sklearn.metrics import mean_squared_error as MSE
# %matplotlib inline

import time

"""# **UTILS**

### *UTILS GENERICAS*
"""

## utils_DP

import scipy.io
import numpy as np
import os
import pandas as pd
import h5py
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split

###      ------      ###
def f_Get_Impact_Database_XY( data_dir ):
    """
    Load .mat file and extract fields
    Args:
        data_dir (strign): Path to X.mat and Y.csv (also contains Y.mat).
    Return:
        X (dataframe): X matrix converted to dataframe, more easy to process data with df.
        Y (dataframe): multiple targets.
    """
    # Data
    file = h5py.File(data_dir + '/X_Scaled.mat', 'r')
    X = np.transpose( file['X_Scaled'][()] )  # file['X'].value
    X = pd.DataFrame( X )
    # Labels
    Y = pd.read_csv( data_dir + '/Y_Scaled.csv', delimiter=';' )

    return X, Y

###      ------      ###

class C_Kfold_idx():
    """
        Class wich contains the index of test, training and validation samples
    """
    def __init__(self):
        self.training = []
        self.test = []
        self.validation = []
    def append_idx(self, att, idx):
        self.__dict__[att].append(idx)
    def get_idx(self, i):
        return self.training[i], self.test[i], self.validation[i]

def Split_kTTV_Dataset(df, train_per=0.5, test_per=0.25, val_per=0.25, kfolds=4):
    """
    Function that generates the row index that will be used for training, test and validation
    Args:
        df (dataframe): Data or labels dataframe.
        x_per (real<1): Percentaje of the dataset that will be used to each task.
        kfolds (integer): Number of different splits combinations.
    Return:
        kTTV_Dataset (C_Kfold_idx obj): Object with the index.
    """
    if np.round(test_per + train_per + val_per, 3) != 1.0:
        print( "Train, test and validation percentajes don't sum 1" )
    idx_compleate = np.linspace( 0, len(df.index)-1, len(df.index) )
    kTTV_Dataset = C_Kfold_idx()
    for k in range(kfolds):
        # Generate training index
        idx_train, idx_tv = train_test_split(
            idx_compleate,
            test_size=(1-train_per),
            random_state=1*k
        )
        # Generate test and validation index
        idx_val, idx_test = train_test_split(
            idx_tv,
            test_size=test_per/(test_per+val_per),
            random_state=1*k
        )
        # Save the index in it's objet attributes
        kTTV_Dataset.append_idx( 'training', idx_train.tolist() )
        kTTV_Dataset.append_idx( 'test', idx_test.tolist() )
        kTTV_Dataset.append_idx( 'validation', idx_val.tolist() )

    return kTTV_Dataset


## utils_DT

from sklearn.preprocessing import MinMaxScaler
from pickle import dump, load
import pandas as pd
import numpy as np


class MinMaxScaleAll:
    def __init__(self) -> None:
        self.min = []
        self.min = []
        pass

    def Fit(self, val):
        # val = df.values
        self.min = np.min(val)
        self.max = np.max(val)

    def Scale(self, val):
        # val = df.values
        valnorm = ( val - self.min )/( self.max - self.min )
        return valnorm # pd.DataFrame(valnorm)

    def Inverse(self, valnorm):
        # valnorm = df.values
        val = valnorm*( self.max - self.min )  + self.min
        return val # pd.DataFrame(val)

def Normalize(x_train, y_train, x_val, y_val, x_test, y_test):
    # Fit the transform
    transX = MinMaxScaleAll()
    transX.Fit(x_train)
    # Normalize X
    x_train = transX.Scale(x_train)
    x_test = transX.Scale(x_test)
    x_val = transX.Scale(x_val)

    # Normalization of targets
    transY = MinMaxScaler( feature_range=(0,1) )
    transY.fit( y_train )
    y_train = pd.DataFrame(
        transY.transform( y_train )
        )
    y_test = pd.DataFrame(
        transY.transform( y_test )
        )
    y_val = pd.DataFrame(
        transY.transform( y_val )
        )

    # Save transforms
    # dump(transX, open('/content/drive/MyDrive/MUSE/S3/CE2/Autoencoder/10/Transforms/transX.pkl', 'wb'))
    # dump(transY, open('/content/drive/MyDrive/MUSE/S3/CE2/Autoencoder/10/Transforms/transY.pkl', 'wb'))
    # Load transform
    # transY = load(open('Transforms/transY.pkl', 'rb'))
    return x_train, y_train, x_val, y_val, x_test, y_test

## Remove wrong impacs

"""### *UTILS PARA USAR XY - mhe*

#### DATA PREPARATION
"""

def f_Data_Preparation():
    # Read dataset
    data_dir = r'/content/drive/MyDrive/MUSE/S3/CE2/Data'
    X, Y = f_Get_Impact_Database_XY( data_dir )
    print("X shape: ", X.shape)

    # Generate the k splits of datasets
    """ Here dataset is splitted in train, validation and test
    minding RAM optimization"""

    ## DATLOADER

    """class Impacts(Dataset):"""

    ## DATASETS
    ds_train = Impacts(x_train,
                    y_train,
                    target=target
    )
    ds_test = Impacts(x_test,
                    y_test,
                    target=target
    )
    ds_val = Impacts(x_val,
                    y_val,
                    target=target
    )


    ## DATALOADER

    dl_train = DataLoader(dataset=ds_train , batch_size=32,
                            shuffle=True)
    dl_test = DataLoader(dataset=ds_test , batch_size=32,
                            shuffle=True)
    dl_val = DataLoader(dataset=ds_val , batch_size=32,
                            shuffle=True)


    ## CHECK DATASET AND DATALOADER

    return ds_train, ds_val, ds_test, dl_train, dl_val, dl_test

"""
#### NET
"""

# Layer classes
""" Layer definitions were here """

"""#### Training"""

""" Training funcitions were here """


"""
# **Data preparation**
"""


"""
# **MODEL**
"""



"""
# **Training**
"""


"""
# **SAVE MODEL**
"""

torch.save(
    model.state_dict(),
    '/content/drive/MyDrive/MUSE/S3/CE2/Redes/Multiple_Outputs/XY/Models/XY_model_k-0'+str(k)
)

"""# **RESULTS**"""

if study == 'XY':
    e_X = np.zeros(len(ds_test))
    e_Y = np.zeros(len(ds_test))
    X_ae = 0
    Y_ae = 0

    with torch.no_grad():
        model.eval()
        for i in range(0,len(ds_test)):
            # Extract the sample
            x, y = ds_test[i]
            y = y.to('cpu').detach().numpy()
            # Inference
            X, Y = model(x.reshape(1,2000, -1).to(device))
            X, Y = X.to('cpu').detach().numpy(), Y.to('cpu').detach().numpy()
            # Error
            e_X[i] = X - y[0]
            e_Y[i] = Y - y[1]

    plt.hist(e_X, bins=20)
    plt.show()
    print(f'Error absoluto acumulado X: {sum(abs(e_X)):.4f}')
    plt.hist(e_Y, bins=20)
    plt.show()
    print(f'Error absoluto acumulado Y: {sum(abs(e_Y)):.4f}')
else:
    e_m = np.zeros(len(ds_test))
    e_h = np.zeros(len(ds_test))
    e_e = np.zeros(len(ds_test))
    m_ae = 0
    h_ae = 0
    e_ae = 0
    with torch.no_grad():
        model.eval()
        for i in range(0,len(ds_test)):
            # Extract the sample
            x, y = ds_test[i]
            y = y.to('cpu').detach().numpy()
            # Inference
            m, h, e = model(x.reshape(1,2000, -1).to(device))
            m, h, e = m.to('cpu').detach().numpy(), h.to('cpu').detach().numpy(), e.to('cpu').detach().numpy()
            # Error
            e_m[i] = m - y[0]
            e_h[i] = h - y[1]
            e_e[i] = e - y[2]

    plt.hist(e_m, bins=20)
    plt.show()
    print(f'Error absoluto acumulado m: {sum(abs(e_m)):.4f}')
    plt.hist(e_h, bins=20)
    plt.show()
    print(f'Error absoluto acumulado h: {sum(abs(e_h)):.4f}')
    plt.hist(e_e, bins=20)
    plt.show()
    print(f'Error absoluto acumulado e: {sum(abs(e_e)):.4f}')

wng_idx = np.where(abs(e_X) == max(abs(e_X)))[0][0]
wrong_impact = ds_test[wng_idx][0].detach().numpy().reshape(2000, -1)
plt.plot(wrong_impact)
plt.show()
ds_test[wng_idx][1]
idx_test[wng_idx]

wng_idx = np.where(abs(e_Y) == max(abs(e_Y)))[0][0]
wrong_impact = ds_test[wng_idx][0].detach().numpy().reshape(2000, -1)
plt.plot(wrong_impact)

pred = np.zeros((len(ds_test), 2))
targ = np.zeros((len(ds_test), 2))
for i in range(0,len(ds_test)):
    # Get single impact
    x, y = ds_test[i]
    # Transform y tensor to numpy
    y = y.cpu().detach().numpy()
    # Inference
    X, Y= model(
        x.reshape(1,2000,8).to(device)
    )
    X, Y = X.cpu().detach().numpy()[0], Y.cpu().detach().numpy()[0]
    # Resize XY and load at pred/target matrix
    targ[i,:] = [y[0]*630, y[1]*710]
    pred[i,:] = [X*630, Y*710]

# Save results
np.savetxt('/content/drive/MyDrive/MUSE/S3/CE2/Redes/Multiple_Outputs/XY/Results/Target_XY_k-0'+str(k)+'.dat', targ, delimiter=',')
np.savetxt('/content/drive/MyDrive/MUSE/S3/CE2/Redes/Multiple_Outputs/XY/Results/Predictions_XY_k-0'+str(k)+'.dat', pred, delimiter=',')
print("CORRECTLY SAVED")

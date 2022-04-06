import pandas as pd
import random

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../0_sources/PYTHON'))
from IA import IA

dataset = 'CM3'
type = 'Correlation'
path_to_dataset = f'../../../../Datasets/{dataset}/3_DATASET/dataset.pkl'
path_to_dataset1 = f'/mnt/sda/0_Andres/1_Universidad/Beca_SHM/98_data/0_CALIBRACION/{dataset}/3_DATASET/dataset.pkl'
#path_to_dataset1 = f'./datasets/{type}/{dataset}.pkl'

dataset = 'CT2'
type = 'Correlation'
path_to_dataset = f'../../../../Datasets/{dataset}/3_DATASET/dataset.pkl'
path_to_dataset2 = f'/mnt/sda/0_Andres/1_Universidad/Beca_SHM/98_data/0_CALIBRACION/{dataset}/3_DATASET/dataset.pkl'
#path_to_dataset2 = f'./datasets/{type}/{dataset}.pkl'


# %%

""" Load dataset """


IA_obj = IA('./models',name='modelos')

#IA_obj.load_datasets([path_to_dataset2],plot_histogram=True,plot_preprocessing=False); IA_obj.save()
IA_obj.load_datasets([path_to_dataset1,path_to_dataset2]); IA_obj.save()

# %%

""" Take a sample """


# %%

""" Import linear regression models """
import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor

def linear_regression(x,a,b):
     return a*x + b

def accuracy(xdata, ydata):
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(linear_regression, xdata, ydata)
    residuals = ydata- linear_regression(xdata, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ydata-np.mean(ydata))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return popt,pcov,r_squared


def evaluate_model(model, X, Y, target='', color='tab:blue'):
    """ Function to evaluate the model """
    if target == 'T':
        idx = 0
        title = f'{str(model)} for Temperature prediction'
    elif target == 'E':
        idx = 1
        title = f'{str(model)} for Deformation prediction'

    model.fit(X['train'], Y['train'][:,idx])
    Y_pred = model.predict(X['test'])

    # Get the relation
    popt,pcov,r_squared = accuracy(Y['test'][:,idx],Y_pred)

    # Plot
    import matplotlib.pyplot as plt
    plt.figure()
    plt.title(title)
    plt.scatter(Y['test'][:,idx],Y_pred,label='Val')
    plt.plot(Y['test'], np.array(linear_regression(Y['test'][:,idx],*popt)),
             label= f'y = {popt[0]:.2f} x +{popt[1]:.2f} | r = {r_squared:.2f}')
    plt.xlabel('Y_test')
    plt.ylabel('Y_pred')
    plt.legend()
    plt.grid()
    #plt.savefig(f'{model}_temperatures.png')
    plt.show()


# %%

""" Evaluate models """

print('*** Evaluating models for temperature and deformation ***')

model_T = GradientBoostingRegressor()
evaluate_model(model_T,IA_obj.X,IA_obj.Y,'T')

model_E = KNeighborsRegressor()
evaluate_model(model_E,IA_obj.X,IA_obj.Y,'E')

print('Done!')

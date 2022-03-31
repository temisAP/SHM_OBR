import pandas as pd
import random

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../0_sources/PYTHON'))
from IA.load_dataset import load_mega_dataset

# %%

""" Load dataset """

info_file = '../1_data/dataset_information.csv'
info_df = pd.read_csv(info_file)
#X,Y = load_mega_dataset(info_df,sets=['CM2','CT2'],leaps=[1,200])
X,Y = load_mega_dataset(info_df,sets=['CM2'],leaps=[1])


# %%

""" Import linear regression models """
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

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


def evaluate_model(model, X, Y, color='tab:blue'):
    """ Function to evaluate the model """
    # Split data in train and test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    # Fit model
    model.fit(X_train, Y_train)
    # Make predictions
    Y_pred = model.predict(X_test)

    # Get the relation
    popt,pcov,r_squared = accuracy(Y_test,Y_pred)

    # Plot
    import matplotlib.pyplot as plt
    plt.figure()
    plt.title(str(model))
    plt.scatter(Y_test,Y_pred,label='Val')
    plt.plot(Y_test, np.array(linear_regression(Y_test,*popt)),
             label= f'y = {popt[0]:.2f} x +{popt[1]:.2f} | r = {r_squared:.2f}')
    plt.xlabel('Y_test')
    plt.ylabel('Y_pred')
    plt.legend()
    plt.grid()
    #plt.savefig(f'{model}_temperatures.png')
    plt.show()

#models = [LinearRegression(), Ridge(), Lasso(), ElasticNet(), SGDRegressor(), SVR(), KNeighborsRegressor(), DecisionTreeRegressor(), RandomForestRegressor(), AdaBoostRegressor(), GradientBoostingRegressor(), MLPRegressor(), GaussianProcessRegressor()]
models = [LinearRegression(), KNeighborsRegressor(), DecisionTreeRegressor(), GradientBoostingRegressor(), MLPRegressor() ]

# %%

""" Normalize data """

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)

# %%

""" Evaluate models """

print('*** Evaluate models for deformation ***')

for model in models:
    print(model)
    evaluate_model(model, X, Y[:,0])
    print("\n")

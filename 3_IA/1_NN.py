import pandas as pd
import random

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../0_sources/PYTHON'))
from IA import IA

# %%

""" Path to dataset """

dataset = 'test_1'
path_to_dataset = f'/mnt/sda/0_Andres/1_Universidad/Beca_SHM/98_data/0_CALIBRACION/{dataset}/3_DATASET/dataset.pkl'

# %%

""" Load dataset """


IA_obj = IA('./models',name='modelos')

IA_obj.load_datasets([path_to_dataset],val_percentage=0); IA_obj.save()
#IA_obj.load_datasets([path_to_dataset1,path_to_dataset2],plot_histogram=True,plot_preprocessing=False); IA_obj.save()

# %%

""" Set models """


from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPRegressor

models = [
    KNeighborsRegressor(),
    GradientBoostingRegressor(),
    RandomForestRegressor(),
    KNeighborsClassifier(),
    GradientBoostingClassifier(),
    RandomForestClassifier(),
    LinearRegression(),
    Ridge(),
    Lasso(),
    ElasticNet(),
    SGDRegressor(),
    SVR(),
    LinearSVR(),
    MLPRegressor()
]

for model in models:

    try:

        IA_obj.model_T = model
        IA_obj.model_E = model

        # %%

        """ Train models """

        print(f'*** Training {model} ***')


        IA_obj.fit_data(representation=False,save=model)

        # %%

        """ Evaluate models """

        IA_obj.load_model()
        IA_obj.save()
        IA_obj.results(representation=False,save=model)

    except Exception as e:
        print(e)



print('Done!')

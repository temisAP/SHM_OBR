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

IA_obj.model_T = KNeighborsRegressor(n_neighbors=10)
IA_obj.model_E = KNeighborsRegressor(n_neighbors=10)

# %%

""" Train models """

print('*** Training models ***')


#IA_obj.fit_data()

# %%

""" Evaluate models """

IA_obj.load_model()
IA_obj.save()
IA_obj.results()



print('Done!')

import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../0_sources/PYTHON'))
from DATASETS import DATASETS

dataset = 'test_1'
path_to_folder = f'/mnt/sda/0_Andres/1_Universidad/SHM/98_data/0_Datasets/{dataset}'
#path_to_folder = f'../../../../Datasets/{dataset}'
#path_to_folder = rf'C:\Users\Luna\Desktop\Andres\Data\{dataset}'
#path_to_folder = f'../../../..{dataset}'

CM2_obj = DATASETS(path_to_folder)
dataset = CM2_obj.load_dataset()


# %%

for i in range(len(dataset.Y)):
    yy = dataset.Y[i]
    T = yy[0]
    E = yy[1]

    eps_the = 24 * T
    eps_mec = E - eps_the
    E = eps_the - eps_mec

    dataset.Y[i] = np.array([T,E])

# %%

dataset.save()

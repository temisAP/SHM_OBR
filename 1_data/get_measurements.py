import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../0_sources/PYTHON'))
from DATASETS import DATASETS

dataset = 'test_1'
path_to_folder = f'/mnt/sda/0_Andres/1_Universidad/SHM/98_data/0_Datasets/{dataset}'

CM2_obj = DATASETS(path_to_folder)

# %%

#CM2_obj.obr()
CM2_obj.obr2measures()
CM2_obj.save()

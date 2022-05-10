import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../0_sources/PYTHON'))
from DATASETS import DATASETS

dataset = 'test_1'
path_to_folder = f'/mnt/sda/0_Andres/1_Universidad/Beca_SHM/98_data/0_CALIBRACION/{dataset}'


CM2_obj = DATASETS(path_to_folder)
CM2_obj.obr_filters('0_mm_20_grados.obr','12.447_mm_50_grados.obr',delta=400)

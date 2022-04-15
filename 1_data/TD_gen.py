import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../0_sources/PYTHON'))
from DATASETS import DATASETS

dataset = 'test_1'
path_to_folder = f'/mnt/sda/0_Andres/1_Universidad/Beca_SHM/98_data/0_CALIBRACION/{dataset}'
#path_to_folder = f'../../../../Datasets/{dataset}'
#path_to_folder = rf'C:\Users\Luna\Desktop\Andres\Data\{dataset}'
#path_to_folder = f'../../../..{dataset}'

CM2_obj = DATASETS(path_to_folder)
#CM2_obj.save()

# %%

CM2_obj.obr()
#CM2_obj.obr_ss('7_mm_22_grados.obr',type='flecha',eps=True)
#CM2_obj.obr_ss('0_mm_20_grados.obr',type='flecha',eps=False)
#CM2_obj.computeOBR()
#exit()
CM2_obj.save()
#exit()

CM2_obj.obr2slices()
CM2_obj.save()

# slices = CM2_obj.load_slices()


CM2_obj.slices2dataset(matches=100,percentage=100,conserve_segment=[100, 300])
CM2_obj.save()

CM2_obj.dataset_plot()
exit()

#dataset = CM2_obj.load_dataset()

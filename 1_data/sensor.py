import sys
import os
import matplotlib.pyplot as plt

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
CM2_obj.obr_TE('0_mm_20_grados.obr',['0_mm_30_grados.obr'],delta=300,type='flecha',eps=True);plt.suptitle('0_mm_30_grados.obr')
CM2_obj.obr_TE('0_mm_20_grados.obr',['12.447_mm_20_grados.obr'],delta=300,type='flecha',eps=True);plt.suptitle('12_mm_20_grados.obr')


dataset = 'CM2'
path_to_folder = f'/mnt/sda/0_Andres/1_Universidad/Beca_SHM/98_data/0_CALIBRACION/{dataset}'
#path_to_folder = f'../../../../Datasets/{dataset}'
#path_to_folder = rf'C:\Users\Luna\Desktop\Andres\Data\{dataset}'
#path_to_folder = f'../../../..{dataset}'

CM2_obj = DATASETS(path_to_folder)
CM2_obj.obr_TE('0_mm_22_grados.obr',['0_mm_53_grados.obr'],delta=300,type='flecha',eps=True);plt.suptitle('0_mm_53_grados.obr')
CM2_obj.obr_TE('0_mm_22_grados.obr',['7_mm_22_grados.obr'],delta=300,type='flecha',eps=True);plt.suptitle('7_mm_22_grados.obr')
CM2_obj.obr_TE('0_mm_22_grados.obr',['7_mm_46_grados.obr'],delta=300,type='flecha',eps=True);plt.suptitle('7_mm_46_grados.obr')

plt.show()

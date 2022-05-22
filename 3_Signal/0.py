import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../0_sources/PYTHON'))
from DATASETS import DATASETS

dataset = 'test_1'
path_to_folder = f'/mnt/sda/0_Andres/1_Universidad/SHM/98_data/0_Datasets/{dataset}'

CM2_obj = DATASETS(path_to_folder)
#CM2_obj.save()

# %%

sample_files = ['0_mm_20_grados.obr','0_mm_30_grados.obr','0_mm_40_grados.obr',
                '12.447_mm_30_grados.obr','6.223_mm_20_grados.obr','12.447_mm_20_grados.obr']

CM2_obj.birds(sample_files)

#CM2_obj.obr()
#CM2_obj.obr_ss('7_mm_22_grados.obr',type='flecha',eps=True)
#CM2_obj.local_analysis(['0_mm_30_grados.obr','0_mm_50_grados.obr','12.447_mm_30_grados.obr','12.447_mm_50_grados.obr'],position = 5.0)

#CM2_obj.local_analysis(['0_mm_30_grados.obr','0_mm_30_grados.obr'],position = 5.1)
#CM2_obj.local_analysis(['0_mm_30_grados.obr','12.447_mm_50_grados.obr'],position = 5.1)
#CM2_obj.local_analysis(['12.447_mm_50_grados.obr','0_mm_30_grados.obr'],position = 5.1)

CM2_obj.save()
#exit()

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../0_sources/PYTHON'))
from DATASETS import DATASETS

dataset = 'test_1'
path_to_folder = f'/mnt/sda/0_Andres/1_Universidad/SHM/98_data/0_Datasets/{dataset}'

CM2_obj = DATASETS(path_to_folder)
#CM2_obj.save()

# %%

sample_files = ['0_mm_20_grados','0_mm_30_grados','0_mm_40_grados',
                '12.447_mm_30_grados','6.223_mm_20_grados','12.447_mm_20_grados']

sf = [sample_files[i] for i in [0,1,2]]

CM2_obj.Stokes_Mueller(sample_files[0],sample_files,limit1=4.8,limit2=5.3,delta=200)
#CM2_obj.Stokes_Mueller(sample_files[0],sf,limit1=5.01,limit2=5.05,delta=1000)
#CM2_obj.Stokes_Mueller(sample_files[0],sample_files[0:2],limit1=4.8,limit2=5.3,delta=1000)

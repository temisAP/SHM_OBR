import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../0_sources/PYTHON'))
from DATASETS import DATASETS
from UTILS.read_obr import multi_read_obr
from SIGNAL.NonlinearCorrections import NonlinearCorrections

dataset = 'test_1'
path_to_folder = f'/mnt/sda/0_Andres/1_Universidad/SHM/98_data/0_Datasets/{dataset}'

CM2_obj = DATASETS(path_to_folder)


sample_files = ['0_mm_20_grados','0_mm_30_grados','0_mm_40_grados',
                '12.447_mm_30_grados','6.223_mm_20_grados','12.447_mm_20_grados']

sample_files = [sample_files[i] for i in [0,3]]

CM2_obj.Corrections(sample_files[0],sample_files,limit1=-0.2,limit2=5.3,delta=1000)

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../0_sources/PYTHON'))
from IA import IA


dataset = 'CM2'
path_to_dataset = f'../../../../Datasets/{dataset}/3_DATASET/dataset.pkl'
path_to_dataset1 = f'/mnt/sda/0_Andres/1_Universidad/Beca_SHM/98_data/0_CALIBRACION/{dataset}/3_DATASET/dataset.pkl'
path_to_dataset1 = f'./datasets/{dataset}.pkl'

dataset = 'CT2'
path_to_dataset = f'../../../../Datasets/{dataset}/3_DATASET/dataset.pkl'
path_to_dataset2 = f'/mnt/sda/0_Andres/1_Universidad/Beca_SHM/98_data/0_CALIBRACION/{dataset}/3_DATASET/dataset.pkl'
path_to_dataset2 = f'./datasets/{dataset}.pkl'


IA_obj = IA('./models',name='FFT')

#IA_obj.load_datasets([path_to_dataset1,path_to_dataset2],plot_histogram=True,plot_preprocessing=False); IA_obj.save()
#IA_obj.load_datasets([path_to_dataset1,path_to_dataset2]); IA_obj.save()

IA_obj.fit_data(15,lr=1e-7); IA_obj.save(); IA_obj.save_model()

IA_obj.results(); IA_obj.save(); IA_obj.save_model()#

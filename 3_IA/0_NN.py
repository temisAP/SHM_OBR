import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../0_sources/PYTHON'))
from IA import IA


dataset = 'CMT_reduced'
path_to_dataset = f'../../../../Datasets/{dataset}/3_DATASET/dataset.pkl'
path_to_dataset = f'/mnt/sda/0_Andres/1_Universidad/Beca_SHM/98_data/0_CALIBRACION/{dataset}/3_DATASET/dataset.pkl'

IA_obj = IA('./models',name=dataset)

#IA_obj.load_datasets(path_to_dataset,plot_histogram=True,plot_preprocessing=True)
#IA_obj.load_datasets(path_to_dataset)
#IA_obj.save()
IA_obj.fit_data(25)
IA_obj.save()
IA_obj.save_model()

IA_obj.results()
IA_obj.save()
IA_obj.save_model()
#

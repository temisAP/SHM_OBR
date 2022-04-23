import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../0_sources/PYTHON'))
from IA import IA
from IA.model import TE


dataset = 'test_1'
type = 'Correlation'
path_to_dataset = f'../../../../Datasets/{dataset}/3_DATASET/dataset.pkl'
path_to_dataset = f'/mnt/sda/0_Andres/1_Universidad/Beca_SHM/98_data/0_CALIBRACION/{dataset}/3_DATASET/dataset.pkl'
#path_to_dataset2 = f'./datasets/{type}/{dataset}.pkl'


IA_obj = IA('./models',name='prueba2')

IA_obj.model_T = TE()

IA_obj.load_datasets([path_to_dataset],plot_histogram=False); IA_obj.save()
#IA_obj.load_datasets([path_to_dataset]); IA_obj.save()

IA_obj.fit_data(100,lr=1e-6); IA_obj.save(); IA_obj.save_model()

IA_obj.results(histograms=True); IA_obj.save(); IA_obj.save_model()

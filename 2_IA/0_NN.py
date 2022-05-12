import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../0_sources/PYTHON'))
from IA import IA
from IA.model import TE


dataset = 'test_1'
path_to_dataset = f'/mnt/sda/0_Andres/1_Universidad/Beca_SHM/98_data/0_CALIBRACION/{dataset}/3_DATASET/dataset_complete.pkl'
#path_to_dataset2 = f'./datasets/{type}/{dataset}.pkl'


IA_obj = IA('./models',name='IA')
#IA_obj.save_model()
#exit()

#IA_obj.model = TE()

#IA_obj.load_datasets([path_to_dataset],plot_histogram=False); IA_obj.save()
#IA_obj.load_datasets([path_to_dataset]); exit(); IA_obj.save()

#IA_obj.fit_data(100,lr=1e-6); IA_obj.save(); IA_obj.save_model()

IA_obj.load_model()
IA_obj.load_scalers()
#IA_obj.results(confusion=True);
#IA_obj.save_model()
IA_obj.save_IA()

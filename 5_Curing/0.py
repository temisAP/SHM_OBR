import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../0_sources/PYTHON'))
from DATASETS import DATASETS

dataset = 'Curado_asimetrico_2'
path_to_folder = f'/mnt/sda/0_Andres/1_Universidad/SHM/98_data/0_Curados/{dataset}'

CM2_obj = DATASETS(path_to_folder)
CM2_obj.load_IA_obj('IA.pkl')
#CM2_obj.take_a_look('0',['20','50','90'],limit1=1,limit2=3,delta=200,val='ss')
#exit()
#CM2_obj.fiber_distribution()

files = list(CM2_obj.obrfiles.keys())
#files.remove('101.obr')
#CM2_obj.measures = None
CM2_obj.curing_evol1D(REF='0.obr',files=files,points=[2,2.25,2.75,3],val='ss')
#CM2_obj.obr2measures()

# %%

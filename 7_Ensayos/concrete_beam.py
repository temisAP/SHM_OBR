import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../0_sources/PYTHON'))
from DATASETS import DATASETS
from IA import IA


dataset = 'ET'
path_to_folder = f'/media/temis/Seagate Backup Plus Drive/Andres/2_Ensayos/{dataset}'

print('\n'+dataset)

CM2_obj = DATASETS(path_to_folder)

obrREF = '0.obr'

# IA loading

if False:

    CM2_obj.IA_obj = IA(CM2_obj.path,name='IA')
    CM2_obj.load_IA_obj('IA.pkl')
    CM2_obj.save()

# For OBR reading

if False:
    #CM2_obj.genCONDITIONStemplate()
    CM2_obj.obr()
    CM2_obj.computeOBR()
    CM2_obj.save()
    exit()


# 1D evolution of available datasets

base_marker = 'o'
markermap = ['-o','-v','-D','-X','-^','-s','-P']

#CM2_obj.take_a_look('Evo1_10',['Evo1_20','Evo1_40.7','Evo1_107.6'],limit1=5.3,limit2=16,delta=500,val='ss')
#CM2_obj.curing_evol1D(REF='0.obr',val='def',t='',colorIS='Load\n[kN]')
CM2_obj.curing_evol1D(val='def',base_marker=base_marker,markermap =markermap,t='kN',colorIS='Load\n[kN]')
#CM2_obj.curing_evol1D(REF='0.obr',points=[],val='ss')

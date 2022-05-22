import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../0_sources/PYTHON'))
from DATASETS import DATASETS
from IA import IA

dataset = 'Curado_equilibrado_2'
path_to_folder = f'/mnt/sda/0_Andres/1_Universidad/SHM/98_data/0_Curados/{dataset}'

CM2_obj = DATASETS(path_to_folder)


# IA loading

if False:

    CM2_obj.IA_obj = IA(CM2_obj.path,name='IA')
    CM2_obj.load_IA_obj('IA.pkl')
    CM2_obj.save()

# For OBR reading

if False:
    #CM2_obj.obr()
    CM2_obj.computeOBR()
    CM2_obj.save()
    exit()


# 1D evolution of available datasets


if dataset == 'Curado_asimetrico_1':

    CM2_obj.take_a_look('0',['20','50','90'],limit1=2,limit2=4,delta=200,val='def')
    files = list(CM2_obj.obrfiles.keys())
    files.remove('101.obr')
    CM2_obj.curing_evol1D(REF='0.obr',files=files,points=[2.25,2.5,2.87],val='ss')

elif dataset == 'Curado_asimetrico_2':

    #CM2_obj.take_a_look('0',['20','50','90'],limit1=1.75,limit2=4,delta=200,val='ss')
    CM2_obj.curing_evol1D(REF='0.obr',points=[2,2.25,2.75,2.94],val='ss')

elif dataset == 'Curado_simetrico_2':

    CM2_obj.take_a_look('0',['20','50','90'],limit1=1,limit2=3,delta=200,val='ss')
    CM2_obj.curing_evol1D(REF='0.obr',points=[2,2.25,2.75,2.94],val='ss')

elif dataset == 'Curado_equilibrado_2':

    #CM2_obj.take_a_look('0',['20','50','90'],limit1=1,limit2=3,delta=200,val='ss')
    CM2_obj.curing_evol1D(REF='0.obr',points=[2,2.25,2.75,2.94],val='ss')

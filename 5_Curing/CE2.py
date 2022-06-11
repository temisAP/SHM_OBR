import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../0_sources/PYTHON'))
from DATASETS import DATASETS
from IA import IA

dataset = 'asimetrico'
#path_to_folder = f'/mnt/sda/0_Andres/1_Universidad/SHM/98_data/0_Curados/{dataset}'
path_to_folder = f'/media/temis/Seagate Backup Plus Drive/Andres/0_Curados/CE2/{dataset}'

CM2_obj = DATASETS(path_to_folder)


# IA loading

if False:

    CM2_obj.IA_obj = IA(CM2_obj.path,name='IA')
    CM2_obj.load_IA_obj('IA.pkl')
    CM2_obj.save()

# For OBR reading

if False:
    #CM2_obj.genCONDITIONStemplate()
    #CM2_obj.obr()
    CM2_obj.computeOBR()
    CM2_obj.save()
    exit()


# 1D evolution of available datasets


if dataset == 'asimetrico':

    #CM2_obj.take_a_look('0',['60','120','180'],limit1=1.8,limit2=4.10,delta=500,val='ss')
    CM2_obj.curing_evol1D(REF='0.obr',val='ss')
    #CM2_obj.curing_evol1D(REF='0.obr',points=[2.6,2.9,3.3,3.6,3.9],val='ss')


elif dataset == 'simetrico':

    #CM2_obj.take_a_look('0',['60','120','184'],limit1=1.8,limit2=4.1,delta=500,val='ss')
    CM2_obj.curing_evol1D(REF='0.obr',val='ss')
    #CM2_obj.curing_evol1D(REF='0.obr',points=[2.25,2.6,2.9],val='ss')

elif dataset == 'compensado':

    #CM2_obj.take_a_look('0',['60','120','240'],limit1=1.8,limit2=4.1,delta=500,val='ss')
    CM2_obj.curing_evol1D(REF='0.obr',val='ss')
    #CM2_obj.curing_evol1D(REF='0.obr',points=[2.40,2.65,2.95,3.30,3.60],val='ss')

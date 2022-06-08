import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../0_sources/PYTHON'))
from DATASETS import DATASETS
from IA import IA


dataset = 'Carbono_simetrico'
#path_to_folder = f'/mnt/sda/0_Andres/1_Universidad/SHM/98_data/0_Curados/{dataset}'
path_to_folder = f'/media/temis/Seagate Backup Plus Drive/Andres/0_Curados/TFM/{dataset}'

print('\n'+dataset)

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


if dataset == 'Aluminio_asimetrico':

    #CM2_obj.take_a_look('0',['60','120','180'],limit1=2,limit2=4.6,delta=500,val='ss')
    CM2_obj.curing_evol1D(REF='0.obr',points=[2.3,2.45,2.8,3.10,3.4,3.66,3.95],val='ss')
    # Ready

elif dataset == 'Aluminio_simetrico':

    #CM2_obj.take_a_look('0',['60','120','180'],limit1=2,limit2=4.6,delta=500,val='ss')
    CM2_obj.curing_evol1D(REF='0.obr',points=[2.27,2.6,3,3.4,3.66,3.95,4.45],val='ss')
    # Ready

elif dataset == 'Acero_asimetrico':

    #CM2_obj.take_a_look('0',['60','120','180'],limit1=2,limit2=4.6,delta=500,val='ss')
    CM2_obj.curing_evol1D(REF='0.obr',points=[2.4,2.6,2.8,3.4,3.66,3.95,4.45],val='ss')
    # Ni idea de qué pasa aquí, quizá haya que repetir

elif dataset == 'Acero_simetrico':

    #CM2_obj.take_a_look('0',['60','120','180'],limit1=2,limit2=4.6,delta=500,val='ss')
    CM2_obj.curing_evol1D(REF='0.obr',points=[2.4,2.65,2.95,3.4,3.66,3.95,4.25,4.45],val='ss')
    # Ready

elif dataset == 'Carbono_asimetrico':

    #CM2_obj.take_a_look('0',['60','120','180'],limit1=2,limit2=4.6,delta=500,val='ss')
    CM2_obj.curing_evol1D(REF='0.obr',points=[2.4,2.6,2.8,3.4,3.66,3.95,4.25],val='ss')
    # Este sale bien refachero

elif dataset == 'Carbono_simetrico':

    #CM2_obj.take_a_look('0',['60','120','180'],limit1=2,limit2=4.6,delta=500,val='ss')
    CM2_obj.curing_evol1D(REF='0.obr',val='ss')

    CM2_obj.curing_evol1D(REF='0.obr',points=[2.3,2.7,2.95,3.4,3.66,3.95,4.20],val='ss')
    # También bien refachero

elif dataset == 'Drift4':

    #CM2_obj.take_a_look('0',['60','120','180'],limit1=2,limit2=4.6,delta=500,val='ss')
    CM2_obj.curing_evol1D(REF='0.obr',points=[2.3,2.7,2.95,3.4,3.66,3.95,4.20],val='ss')

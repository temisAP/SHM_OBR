import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../0_sources/PYTHON'))
from DATASETS import DATASETS
from IA import IA

dataset = 'simetrico'
#path_to_folder = f'/mnt/sda/0_Andres/1_Universidad/SHM/98_data/0_Curados/{dataset}'
path_to_folder = f'/media/temis/Seagate Backup Plus Drive/Andres/0_Curados/CE2/{dataset}'

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


if dataset == 'asimetrico':

    #CM2_obj.take_a_look('0',['60','120','180'],limit1=1.8,limit2=4.10,delta=500,val='ss')
    #CM2_obj.curing_evol1D(REF='0.obr',val='ss')
    CM2_obj.curing_evol1D(REF='0.obr',points=[[2.22270236297664, 2.4415436926458973], [2.5719642830548493, 2.7664899094275226], [2.861542204132352, 3.0715414598755792], [3.206383087247546, 3.396487676657204], [3.4826978974360028, 3.7015392271052603]],val='ss')


elif dataset == 'simetrico':

    #CM2_obj.take_a_look('0',['60','120','184'],limit1=1.8,limit2=4.1,delta=500,val='ss')
    CM2_obj.curing_evol1D(REF='0.obr',val='ss')
    #CM2_obj.curing_evol1D(REF='0.obr',points=[[1.8645983689724, 1.9773348115292904], [2.1961761411985483, 2.388491249089714], [2.5520696167212806, 2.7311216137234], [2.8527001302063213, 3.038383682652964], [3.186488420913977, 3.224067235099607]],val='ss')

elif dataset == 'compensado':

    #CM2_obj.take_a_look('0',['60','120','240'],limit1=1.8,limit2=4.1,delta=500,val='ss')
    CM2_obj.curing_evol1D(REF='0.obr',val='ss')
    #CM2_obj.curing_evol1D(REF='0.obr',points=[[2.585227393943895, 2.7952266496871223], [2.8659632410953675, 3.058278348986533], [3.4981715268065563, 3.703749745586768]],val='ss')

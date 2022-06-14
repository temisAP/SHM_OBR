import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../0_sources/PYTHON'))
from DATASETS import DATASETS
from IA import IA

dataset = 'asimetrico'
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
    #CM2_obj.curing_evol1D(REF='0.obr',points=[[2.22270236297664, 2.4415436926458973], [2.5719642830548493, 2.7664899094275226], [2.861542204132352, 3.0715414598755792], [3.206383087247546, 3.396487676657204], [3.4826978974360028, 3.7015392271052603]], val='ss')
    CM2_obj.curing_evol1D(REF='1800.obr',points=[[2.22270236297664, 2.4415436926458973], [2.5719642830548493, 2.7664899094275226], [2.861542204132352, 3.0715414598755792], [3.206383087247546, 3.396487676657204], [3.4826978974360028, 3.7015392271052603]], val='ss')


elif dataset == 'simetrico':

    #CM2_obj.take_a_look('0',['60','120','184'],limit1=1.8,limit2=4.1,delta=500,val='ss')
    #CM2_obj.curing_evol1D(REF='0.obr',val='ss')
    #CM2_obj.curing_evol1D(REF='0.obr',points=[[2.1409131791608567, 2.2956494728663923], [2.3067020652739307, 2.4039648784602674], [2.5454380612767573, 2.7488057615754613], [2.823963389946722, 2.945541906429643], [2.963226054281704, 3.0759624968385944]],val='ss')
    CM2_obj.curing_evol1D(REF='1800.obr',points=[[2.1409131791608567, 2.2956494728663923], [2.3067020652739307, 2.4039648784602674], [2.5454380612767573, 2.7488057615754613], [2.823963389946722, 2.945541906429643], [2.963226054281704, 3.0759624968385944]],val='ss')

elif dataset == 'compensado':

    #CM2_obj.take_a_look('0',['60','120','240'],limit1=1.8,limit2=4.1,delta=500,val='ss')
    #CM2_obj.curing_evol1D(REF='0.obr',val='ss')
    #CM2_obj.curing_evol1D(REF='0.obr',points=[[2.22270236297664, 2.4415436926458973], [2.5719642830548493, 2.7664899094275226], [2.861542204132352, 3.0715414598755792], [3.206383087247546, 3.396487676657204], [3.4826978974360028, 3.7015392271052603]], val='ss')
    CM2_obj.curing_evol1D(REF='1800.obr',points=[[2.22270236297664, 2.4415436926458973], [2.5719642830548493, 2.7664899094275226], [2.861542204132352, 3.0715414598755792], [3.206383087247546, 3.396487676657204], [3.4826978974360028, 3.7015392271052603]], val='ss')

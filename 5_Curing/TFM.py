import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../0_sources/PYTHON'))
from DATASETS import DATASETS
from IA import IA


dataset = 'Carbono_asimetrico'
#path_to_folder = f'/mnt/sda/0_Andres/1_Universidad/SHM/98_data/0_Curados/{dataset}'
path_to_folder = f'/media/temis/Seagate Backup Plus Drive/Andres/0_Curados/TFM/{dataset}'

print('\n'+dataset)

CM2_obj = DATASETS(path_to_folder)

obrREF = '1800.obr'

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
    #CM2_obj.curing_evol1D(REF=obrREF,val='ss')
    CM2_obj.curing_evol1D(REF=obrREF,points=[[2.6706701242283652, 2.8181222997255633], [2.8508894498360524, 3.012384689666317], [3.0498328612211614, 3.1785609509409376], [3.3096295513828915, 3.43601713038049], [3.513253984212356, 3.6724087133204435], [3.7426240349857762, 3.85262803892813], [3.983696639370084, 4.110084218367683]],val='ss')
    # Ready

elif dataset == 'Aluminio_simetrico':

    #CM2_obj.take_a_look('0',['60','120','180'],limit1=2,limit2=4.6,delta=500,val='ss')
    #CM2_obj.curing_evol1D(REF=obrREF,val='ss')
    CM2_obj.curing_evol1D(REF=obrREF,points=[[2.2029814991312295, 2.34041794805481], [2.385397149520709, 2.5128382203407567], [2.882667210171483, 2.9926163693103476], [3.0101082809915303, 3.1200574401303953], [3.314967313149291, 3.484888740909355], [3.959669200827179, 4.099604494276643]],val='ss')


elif dataset == 'Acero_asimetrico':

    #CM2_obj.take_a_look('0',['60','120','180'],limit1=2,limit2=4.6,delta=500,val='ss')
    #CM2_obj.curing_evol1D(REF=obrREF,val='ss')
    CM2_obj.curing_evol1D(REF=obrREF,points=[],val='ss')

    # Ni idea de qué pasa aquí, quizá haya que repetir

elif dataset == 'Acero_simetrico':

    #CM2_obj.take_a_look('0',['60','120','180'],limit1=2,limit2=4.6,delta=500,val='ss')
    #CM2_obj.curing_evol1D(REF=obrREF,val='ss')
    CM2_obj.curing_evol1D(REF=obrREF,points=[[2.280445679433611, 2.4353740400383748], [2.595300089694905, 2.6752631145231702], [2.897660277326783, 3.0525886379315463], [3.242500821898676, 3.434911850391689], [3.5248702533234875, 3.73977088254945], [3.8572165752659644, 4.07461604901781], [4.429451971693236, 4.5943757104015335]],val='ss')


elif dataset == 'Carbono_asimetrico':

    #CM2_obj.take_a_look('0',['60','120','180'],limit1=2,limit2=4.6,delta=500,val='ss')
    #CM2_obj.curing_evol1D(REF=obrREF,val='ss')
    CM2_obj.curing_evol1D(REF='7860.obr',points=[[2.227969944390062, 2.442870573616025], [2.597798934220789, 2.7977064962914513], [2.822694941550284, 3.0400944153021303], [3.4748933628058216, 3.627322878884702], [3.667304391298835, 3.837225819058898], [3.8572165752659644, 3.997151868715428], [4.1670732964754915, 4.2470363213037565]],val='ss')


elif dataset == 'Carbono_simetrico':

    #CM2_obj.take_a_look('0',['60','120','180'],limit1=2,limit2=4.6,delta=500,val='ss')
    #CM2_obj.curing_evol1D(REF=obrREF,val='ss')
    CM2_obj.curing_evol1D(REF=obrREF,points=[[2.123018474302964, 2.415383283831309], [2.5278312874960562, 2.767720361980852], [2.8176972524985175, 3.085073616768029], [3.467396829228172, 3.7047870591870837], [3.777253550437699, 4.029636847551911], [4.112098716906059, 4.269525922036706]],val='ss')

elif dataset == 'Drift4':

    #CM2_obj.take_a_look('0',['60','120','180'],limit1=2,limit2=4.6,delta=500,val='ss')
    CM2_obj.curing_evol1D(REF=obrREF,points=[2.3,2.7,2.95,3.4,3.66,3.95,4.20],val='ss')

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../0_sources/PYTHON'))
from DATASETS import DATASETS
from IA import IA


dataset = 'Acero_simetrico'
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
    #CM2_obj.curing_evol1D(REF='0.obr',val='ss')
    CM2_obj.curing_evol1D(REF='0.obr',points=[[2.375401771417176, 2.5628151108584225], [2.6877573371525867, 2.855179920386767], [2.9701267685773978, 3.160038952544528], [3.337456913882241, 3.539863320478787], [3.614828656255286, 3.822232751903598], [3.9271842219906965, 4.127091784061359], 4.534403441780334],val='ss')
    #CM2_obj.curing_evol1D(REF='0.obr',points=[2.3,2.45,2.8,3.10,3.4,3.66,3.95],val='ss')
    # Ready

elif dataset == 'Aluminio_simetrico':

    #CM2_obj.take_a_look('0',['60','120','180'],limit1=2,limit2=4.6,delta=500,val='ss')
    #CM2_obj.curing_evol1D(REF='0.obr',val='ss')
    CM2_obj.curing_evol1D(REF='0.obr',points=[[2.2029814991312295, 2.34041794805481], [2.385397149520709, 2.5128382203407567], [2.882667210171483, 2.9926163693103476], [3.0101082809915303, 3.1200574401303953], [3.314967313149291, 3.484888740909355], [3.959669200827179, 4.099604494276643]],val='ss')
    #CM2_obj.curing_evol1D(REF='0.obr',points=[2.27,2.6,3,3.4,3.66,3.95,4.45],val='ss')


elif dataset == 'Acero_asimetrico':

    #CM2_obj.take_a_look('0',['60','120','180'],limit1=2,limit2=4.6,delta=500,val='ss')
    #CM2_obj.curing_evol1D(REF='0.obr',val='ss')
    CM2_obj.curing_evol1D(REF='0.obr',points=[],val='ss')
    #CM2_obj.curing_evol1D(REF='0.obr',points=[2.4,2.6,2.8,3.4,3.66,3.95,4.45],val='ss')

    # Ni idea de qué pasa aquí, quizá haya que repetir

elif dataset == 'Acero_simetrico':

    #CM2_obj.take_a_look('0',['60','120','180'],limit1=2,limit2=4.6,delta=500,val='ss')
    #CM2_obj.curing_evol1D(REF='0.obr',val='ss')
    CM2_obj.curing_evol1D(REF='0.obr',points=[[2.280445679433611, 2.4353740400383748], [2.595300089694905, 2.6752631145231702], [2.897660277326783, 3.0525886379315463], [3.242500821898676, 3.434911850391689], [3.5248702533234875, 3.73977088254945], [3.8572165752659644, 4.07461604901781], [4.429451971693236, 4.5943757104015335]],val='ss')
    #CM2_obj.curing_evol1D(REF='0.obr',points=[2.4,2.65,2.95,3.4,3.66,3.95,4.25,4.45],val='ss')

elif dataset == 'Carbono_asimetrico':

    #CM2_obj.take_a_look('0',['60','120','180'],limit1=2,limit2=4.6,delta=500,val='ss')
    #CM2_obj.curing_evol1D(REF='0.obr',val='ss')
    CM2_obj.curing_evol1D(REF='0.obr',points=[[2.215803327762383, 2.39805531706629], [2.6143276777069255, 2.820879932251353], [2.869480462732395, 3.0687426377046663], [3.4818471467935215, 3.715129693102522], [3.7856004623000326, 4.001872822940669], [4.198704971388888, 4.2473055018699295], [4.448997703366253, 4.519468472563764]],val='ss')
    #CM2_obj.curing_evol1D(REF='0.obr',points=[2.4,2.6,2.8,3.4,3.66,3.95,4.25],val='ss')

elif dataset == 'Carbono_simetrico':

    #CM2_obj.take_a_look('0',['60','120','180'],limit1=2,limit2=4.6,delta=500,val='ss')
    #CM2_obj.curing_evol1D(REF='0.obr',val='ss')
    CM2_obj.curing_evol1D(REF='0.obr',points=  [[2.1088701743198897, 2.434271030694304], [2.5644313732440693, 2.7644926404964876], [2.807879421346409, 3.0151718187404803], [3.4827849012340835, 3.697308428769808], [3.7937234973251903, 4.041992298855299], [4.107072470130182, 4.246874319535486], [4.442114833360135, 4.591558189620977]],val='ss')
    #CM2_obj.curing_evol1D(REF='0.obr',points=  [[2.1088701743198897, 2.434271030694304]],val='ss')
    #CM2_obj.curing_evol1D(REF='0.obr',points=[2.3,2.7,2.95,3.4,3.66,3.95,4.20],val='ss')

elif dataset == 'Drift4':

    #CM2_obj.take_a_look('0',['60','120','180'],limit1=2,limit2=4.6,delta=500,val='ss')
    CM2_obj.curing_evol1D(REF='0.obr',points=[2.3,2.7,2.95,3.4,3.66,3.95,4.20],val='ss')

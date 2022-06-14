import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../0_sources/PYTHON'))
from DATASETS import DATASETS
from IA import IA


dataset = 'Aluminio_asimetrico'
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
    CM2_obj.curing_evol1D(REF='8400.obr',points=[[2.6677665809455204, 2.8002053408173344], [2.837688008705584, 2.9951152138362307], [3.0226025036209467, 3.1775308642257105], [3.2999742459939916, 3.4274153168140393], [3.4798910518575883, 3.6498124796176517], [3.6822974584541344, 3.8247315964294817], [3.9821588015601286, 4.09710564975076]],val='ss')

elif dataset == 'Aluminio_simetrico':

    #CM2_obj.take_a_look('0',['60','120','180'],limit1=2,limit2=4.6,delta=500,val='ss')
    #CM2_obj.curing_evol1D(REF=obrREF,val='ss')
    CM2_obj.curing_evol1D(REF='8584.obr',points=[[2.332001257533259, 2.5285976082594677], [2.6431363517260413, 2.808961099729887], [2.870504305174613, 3.0551339215087916], [3.2688256070807578, 3.386783417516483], [3.3919120179702102, 3.4927744935601783], [3.569703500366086, 3.694499444740114]],val='ss')

elif dataset == 'Acero_asimetrico':

    #CM2_obj.take_a_look('0',['60','120','180'],limit1=2,limit2=4.6,delta=500,val='ss')
    #CM2_obj.curing_evol1D(REF=obrREF,val='ss')
    CM2_obj.curing_evol1D(REF=obrREF,points=[],val='ss')

    # Ni idea de qué pasa aquí, quizá haya que repetir

elif dataset == 'Acero_simetrico':

    #CM2_obj.take_a_look('0',['60','120','180'],limit1=2,limit2=4.6,delta=500,val='ss')
    #CM2_obj.curing_evol1D(REF=obrREF,val='ss')
    CM2_obj.curing_evol1D(REF='8400.obr',points=[[2.2629537677524283, 2.4853509305560406], [2.5678127999101887, 2.7727180510326184], [2.87267183206795, 3.080075927716263], [3.249997355476326, 3.4499049175469887], [3.504879497116421, 3.7522651051788665], [4.144583695742542, 4.369479703072038]],val='ss')

elif dataset == 'Carbono_asimetrico':

    #CM2_obj.take_a_look('0',['60','120','180'],limit1=2,limit2=4.6,delta=500,val='ss')
    #CM2_obj.curing_evol1D(REF=obrREF,val='ss')
    CM2_obj.curing_evol1D(REF='7860.obr',points=[[2.227969944390062, 2.442870573616025], [2.597798934220789, 2.7977064962914513], [2.822694941550284, 3.0400944153021303], [3.4748933628058216, 3.627322878884702], [3.667304391298835, 3.837225819058898], [3.8572165752659644, 3.997151868715428], [4.1670732964754915, 4.2470363213037565]],val='ss')

elif dataset == 'Carbono_simetrico':

    #CM2_obj.take_a_look('0',['60','120','180'],limit1=2,limit2=4.6,delta=500,val='ss')
    #CM2_obj.curing_evol1D(REF=obrREF,val='ss')
    CM2_obj.curing_evol1D(REF='7440.obr',points=[[2.123018474302964, 2.415383283831309], [2.5278312874960562, 2.767720361980852], [2.8176972524985175, 3.085073616768029], [3.467396829228172, 3.7047870591870837], [3.777253550437699, 4.029636847551911], [4.112098716906059, 4.269525922036706]],val='ss')

elif dataset == 'Drift4':

    #CM2_obj.take_a_look('0',['60','120','180'],limit1=2,limit2=4.6,delta=500,val='ss')
    CM2_obj.curing_evol1D(REF=obrREF,points=[2.3,2.7,2.95,3.4,3.66,3.95,4.20],val='ss')

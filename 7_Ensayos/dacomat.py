import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../0_sources/PYTHON'))
from DATASETS import DATASETS
from IA import IA


dataset = 'BROKEN'
path_to_folder = f'/media/temis/Seagate Backup Plus Drive/Andres/2_Ensayos/DACOMAT/{dataset}'

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

if dataset == 'BEAM_1':

    #CM2_obj.take_a_look('0',['60','65','70'],limit1=2.55,limit2=8.85,delta=500,val='ss')
    #CM2_obj.curing_evol1D(REF='0.obr',val='def',t='',colorIS='Load\n[kN]')
    CM2_obj.curing_evol1D(val='def',base_marker=base_marker,markermap = markermap ,t='kN',t_limits = [0,165],colorIS='Load\n[kN]')
    #CM2_obj.curing_evol1D(REF='0.obr',points=[],val='ss')

if dataset == 'BEAM_2':

    #CM2_obj.take_a_look('0',['20','30','45'],limit1=10,limit2=16.2,delta=500,val='ss')
    #CM2_obj.curing_evol1D(REF='0.obr',val='def',t='',colorIS='Load\n[kN]')
    CM2_obj.curing_evol1D(val='def',base_marker=base_marker,markermap = markermap ,t='kN',t_limits = [0,165],colorIS='Load\n[kN]')
    #CM2_obj.curing_evol1D(REF='0.obr',points=[],val='ss')

if dataset == 'BEAM_3':

    #CM2_obj.take_a_look('0',['40','50','60'],limit1=9,limit2=15.29,delta=500,val='ss')
    #CM2_obj.curing_evol1D(REF='0.obr',val='def',t='',colorIS='Load\n[kN]')
    CM2_obj.curing_evol1D(val='def',base_marker=base_marker,markermap = markermap ,t='kN',t_limits = [0,165],colorIS='Load\n[kN]')
    #CM2_obj.curing_evol1D(REF='0.obr',points=[],val='ss')

if dataset == 'BEAM_4':

    #CM2_obj.take_a_look('0',['60','65','70'],limit1=2.6,limit2=9.04,delta=500,val='ss')
    #CM2_obj.curing_evol1D(REF='0.obr',val='def',t='',colorIS='Load\n[kN]')
    CM2_obj.curing_evol1D(val='def',base_marker=base_marker,markermap = markermap,t='kN',t_limits = [0,165],colorIS='Load\n[kN]')
    #CM2_obj.curing_evol1D(REF='0.obr',points=[],val='ss')

if dataset == 'BROKEN':

    CM2_obj.take_a_look(['C_0','E_0','ES_0'],['Conventional','Experimental','Experimental_with_stiffeners',],limit1=[10,9,2.6],limit2=[16.2,15.29,9.04],delta=500,val='def',common_origin=True)

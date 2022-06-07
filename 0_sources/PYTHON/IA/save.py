import os
import pickle
import torch
import sys
import numpy as np
from .model import TE_single


def save(self):
    """ Save the object in the root of folder """

    # Change self and its objects to global

    if 'self' in globals():
        pass
    else:
        globals()['self'] = self

    for key in self.__dict__.keys():
        if isinstance(self.__dict__[key], object):
            globals()[key] = self.__dict__[key]

    # Define path

    path_to = os.path.join(self.path,self.name)

    # Save with pickle

    if not os.path.exists(self.path):
        os.makedirs(self.path)

    with open(path_to, 'wb') as outp:
        pickle.dump(self.__dict__, outp, pickle.HIGHEST_PROTOCOL)
    print('')
    print(f'{self.name} saved!')

    # Return self and its objects to local

    if 'self' in globals():
        del globals()['self']
    else:
        pass

    for key in self.__dict__.keys():
        if isinstance(self.__dict__[key], object):
            del globals()[key]
        else:
            pass

def save_model(self):

    path_to = os.path.join(self.path,self.name.replace('.pkl',f'_model.pkl'))

    if isinstance(self.model, torch.nn.Module):
        # Save full model
        torch.save(self.model.state_dict(),path_to)
        torch.save(TE_single(self.model,0).state_dict(),path_to.replace('.pkl','T.pkl'))
        torch.save(TE_single(self.model,1).state_dict(),path_to.replace('.pkl','E.pkl'))

        try:
            input_names  = ['Cross correlations (concatenated)']
            output_names = ['Temperature','Deformation']
            batch = torch.randn(1,12001)
            torch.onnx.export(self.model, batch, path_to.replace('.pkl','.onnx'), input_names=input_names, output_names=output_names)
        except Exception as e:
            print('Error while exporting model in ONNX format')
            print(e)
    else:
        try:
            with open(path_to, 'wb') as outp:
                pickle.dump(self.model.__dict__, outp, pickle.HIGHEST_PROTOCOL)
        except:
            save_keras_model(self.model,self.name.replace('.pkl',f'_model.pkl'),self.path)

    print(f'model saved!')

def save_keras_model(model, model_name, model_dir):
    """ Save a keras model and its weights """
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, model_name + '.json')
    weights_path = os.path.join(model_dir, model_name + '_weights.hdf5')
    options = {'file_arch': model_path,
                'file_weight': weights_path}
    json_string = model.to_json()
    open(options['file_arch'], 'w').write(json_string)
    model.save_weights(options['file_weight'])

def save_scalers(self,path_to=[None,None]):
    print('Under construction')

def save_IA(self,path_to=None):

    from copy import copy
    IA_obj = copy(self); print('error in save_IA') if IA_obj is self else False
    IA_obj.name = self.name.replace('.pkl',f'_IA.pkl')

    # Emtpy dictionaries
    IA_obj.X = dict()
    IA_obj.Y = dict()
    IA_obj.dl = None
    IA_obj.save()

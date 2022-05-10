import os
import pickle5 as pickle
import sys
import numpy as np
import torch
from .model import TE
from .pre_processing import a_scaler

try:
    import keras
    from keras.models import load_model
    from keras.models import model_from_json
except ImportError as e:
    print('Warning: Failed to import keras. Ignore warning message if the model is not keras based')

def load(self):

    """ Load the object existing in the root of the folder """

    path_to = os.path.join(self.path,self.name)

    new_path = self.path
    new_name = self.name

    with open(path_to, 'rb') as inp:
        self.__dict__ = pickle.load(inp)

    self.path = new_path
    self.name = new_name

    return self

def load_model(self,path_to = None):

    path_to = path_to if path_to else os.path.join(self.path,self.name.replace('.pkl',f'_model.pkl'))

    if not hasattr(self, "model") or self.model == 'torch':
        self.model = TE()

    if isinstance(self.model, torch.nn.Module):
        self.model.load_state_dict(torch.load(path_to))

    else:
        try:
            with open(path_to, 'rb') as inp:
                self.model.__dict__ = pickle.load(inp)
        except:
            self.model = load_keras_model(self.name.replace('.pkl',f'_model.pkl'),self.path)

    print(f' model loaded!')

def load_keras_model(model_name, model_dir):
    """ Load a keras model and its weights """
    file_arch = os.path.join(model_dir, model_name + '.json')
    file_weight = os.path.join(model_dir, model_name + '_weights.hdf5')
    # load architecture
    model = model_from_json(open(file_arch).read())
    # load weights
    model.load_weights(file_weight)
    return model

def load_scalers(self,path_to = [None,None]):

    path_to_X = path_to if path_to[0] else os.path.join(self.path,self.name.replace('.pkl',f'_scalerX.pkl'))
    path_to_Y = path_to if path_to[1] else os.path.join(self.path,self.name.replace('.pkl',f'_scalerY.pkl'))

    self.scalerX = a_scaler()
    self.scalerY = a_scaler()

    with open(path_to_X, 'rb') as inp:
        self.scalerX.__dict__ = pickle.load(inp)
    with open(path_to_Y, 'rb') as inp:
        self.scalerY.__dict__ = pickle.load(inp)


    print(f' scalers loaded!')

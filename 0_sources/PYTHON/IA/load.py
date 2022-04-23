import os
import pickle5 as pickle
import sys
import numpy as np
import torch
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

def load_model(self):

    models = {'temperature':self.model_T,'deformation':self.model_E}

    for key,val in models.items():

        path_to = os.path.join(self.path,self.name.replace('.pkl',f'_model_{key}.pkl'))

        if isinstance(val, torch.nn.Module):
            val.load_state_dict(torch.load(path_to))
        else:
            try:
                with open(path_to, 'rb') as inp:
                    val.__dict__ = pickle.load(inp)
            except:
                val = load_keras_model(self.name.replace('.pkl',f'_model_{key}.pkl'),self.path)

        print(f'{key} model loaded!')


def load_keras_model(model_name, model_dir):
    """ Load a keras model and its weights """
    file_arch = os.path.join(model_dir, model_name + '.json')
    file_weight = os.path.join(model_dir, model_name + '_weights.hdf5')
    # load architecture
    model = model_from_json(open(file_arch).read())
    # load weights
    model.load_weights(file_weight)
    return model

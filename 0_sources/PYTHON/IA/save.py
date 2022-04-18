import os
import pickle
import torch
import sys
import numpy as np


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

    models = {'temperature':self.model_T,'deformation':self.model_E}

    for key,val in models.items():

        path_to = os.path.join(self.path,self.name.replace('.pkl',f'_model_{key}.pkl'))

        if isinstance(val, torch.nn.Module):
            torch.save(val.state_dict(),path_to)
        else:
            try:
                with open(path_to, 'wb') as outp:
                    pickle.dump(val.__dict__, outp, pickle.HIGHEST_PROTOCOL)
            except:
                save_keras_model(val,self.name.replace('.pkl',f'_model_{key}.pkl'),self.path)

        print(f'{key} model saved!')


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

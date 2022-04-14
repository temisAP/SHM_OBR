import os
import pickle
import torch

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

    path_to = os.path.join(self.path,self.name.replace('.pkl','_model_T.pkl'))
    print(f'Temperature model saved!')
    torch.save(self.model_y.state_dict(),path_to)

import os
import pickle

def load(self):
    """ Load the object existing in the root of the folder """

    path_to_dataset = os.path.join(self.path,self.name)
    
    new_path = self.path
    new_name = self.name

    with open(path_to_dataset, 'rb') as inp:
        self.__dict__ = pickle.load(inp)
    
    self.path = new_path
    self.name = new_name

    return self

def load_slices(self):

    """ Load slices from their ubication """

    path_to = os.path.join(self.path,self.folders['2_SLICES'])
    name = self.INFO['slices filename']

    if os.path.exists(path_to):
        from .obr2slices import slices
        return slices(path_to,name)
    else:
        print('NO SLICES FOUND')

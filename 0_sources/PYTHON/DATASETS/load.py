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
        print('SLICES FOUND')
    else:
        print('NO SLICES FOUND')

def load_dataset(self):

    """ Load datasets from their ubication """

    path_to = os.path.join(self.path,self.folders['3_DATASET'])
    name = self.INFO['dataset filename']

    if os.path.exists(path_to):
        from .slices2dataset import dataset
        return dataset(path_to,name)
        print('DATASET FOUND')
    else:
        print('NO DATASET FOUND')

def load_obrfiles(self):

    path_to = os.path.join(self.path,self.folders['1_PROCESSED'],self.name.replace('.pkl','_obrfiles.pkl'))

    if os.path.exists(path_to):
        with open(path_to_dataset, 'rb') as inp:
            self.obrfiles.__dict__ = pickle.load(inp)
        print('OBRFILES FOUND')
    else:
        print('NO OBRFILES FOUND')

def load_measures(self):

    path_to = os.path.join(self.path,self.folders['1_PROCESSED'],self.name.replace('.pkl','_measures.pkl'))

    if os.path.exists(path_to):
        with open(path_to_dataset, 'rb') as inp:
            self.measures.__dict__ = pickle.load(inp)
        print('MEASURES FOUND')
    else:
        print('NO MEASURES FOUND')

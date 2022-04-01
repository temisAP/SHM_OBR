import os
import pickle

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

    path_to = os.path.join(self.path,self.name.replace('.pkl','_model_T.pkl'))
    self.model_T.load_state_dict(torch.load(path_to))

    path_to = os.path.join(self.path,self.name.replace('.pkl','_model_E.pkl'))
    self.model_E.load_state_dict(torch.load(path_to))

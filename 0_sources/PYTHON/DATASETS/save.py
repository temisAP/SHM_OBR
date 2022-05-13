import os
import pickle

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

    path_to_dataset = os.path.join(self.path,self.name)

    # Save with pickle

    with open(path_to_dataset, 'wb') as outp:
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

def save_something(self,object_to_save,path_to):

    # Save with pickle
    with open(path_to, 'wb') as outp:
        pickle.dump(object_to_save.__dict__, outp, pickle.HIGHEST_PROTOCOL)
    print('')
    print(object_to_save,'saved in',path_to)

def save_obrfiles(self):

    """ Save obrfiles once computed """

    path_to = os.path.join(self.path,self.folders['1_PROCESSED'],self.name.replace('.pkl','_obrfiles.pkl'))
    object_to_save = self.obrfiles
    self.save_measures(object_to_save,path_to)


def save_measures(self):

    """ Save obrfiles once computed by the model to get measures """

    path_to = os.path.join(self.path,self.folders['1_PROCESSED'],self.name.replace('.pkl','_measures.pkl'))
    object_to_save = self.measures
    self.save_measures(object_to_save,path_to)

import os

def clear_slices(self,auto=True):
    """ Function to clear slices and its book """

    if auto == True or 'n' not in input('Are you sure? (yes/no) '):
        os.remove(os.path.join(self.path,self.folders['2_SLICES'],self.INFO['slices filename']))
        os.remove(os.path.join(self.path,self.folders['4_INFORMATION'],self.INFO['slices book filename']))
    else:
        pass

def clear_dataset(self,auto=True):
    """ Function to clear dataset and its book """

    if auto == True or 'n' not in input('Are you sure? (yes/no) '):
        os.remove(os.path.join(self.path,self.folders['3_DATASET'],self.INFO['dataset filename']))
        os.remove(os.path.join(self.path,self.folders['4_INFORMATION'],self.INFO['dataset book filename']))
    else:
        pass

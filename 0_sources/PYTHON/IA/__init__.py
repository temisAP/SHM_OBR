import torch
import os
from .model import splitter

class IA(object):

    def __init__(self,path=None,name=None,showpath=False):

        if path is None:
            from .PathSelector import PathSelector
            import tkinter as tk
            # Initialize gui
            root = tk.Tk()
            root.geometry("400x100")
            root.title("Path Selector")

            # Create gui
            app = PathSelector(master=root)
            app.pack_propagate(0)
            app.mainloop()

            # Get path
            path = app.path

        if name is None:
            import time
            name = f'model_{time.strftime("%Y%m%d_%H%M%S")}'
            print(f'Model will be named as {name}')

        # In construction
        self.path = path
        self.name = f'{name}.pkl'
        torch.cuda.empty_cache()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(r"Let's do some magic with",self.device)

        if showpath:
             print(os.listdir('.'))

        try:

            self.load()
            print('\nOBJECT FOUND')

        except Exception as e:

            if 'No such file or directory' in str(e):
                print('\nNO OBJECT FOUND IN PATH')
                print('Creating new one \n')
            else:
                print(e)
                exit()

            # Atributes initialization

            torch.manual_seed(0)

            self.X  = dict()          # Dataset inputs           for train, test and validation (valid)
            self.Y  = dict()          # Dataset correct outputs  for train, test and validation (valid)
            self.dl = dict()          # Dataloaders              for train, test and validation (valid)

    from .load_datasets import load_datasets
    from .pre_processing import pre_processing
    from .fit_data import fit_data
    from .results import results

    def clear_dataset(self,auto=False):
        """ Function to clear the dataset """

        if auto == True or 'n' not in input('Are you sure? (yes/no) '):
            self.X = dict()
            self.Y = dict()
        else:
            pass

    from .load import load, load_model
    from .save import save, save_model

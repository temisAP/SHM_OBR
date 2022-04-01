import os
import sys
import pandas as pd


class DATASETS(object):
    """
        Dataset class
        for further information open pdfs attached
    """

    def __init__(self,path=None,showpath=False):

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

        # In construction
        self.path = os.path.abspath(path)
        self.name = f'{os.path.basename(os.path.normpath(path))}.pkl'

        if showpath:
             print(os.listdir(self.path))

        try:
            print('\nOBJECT FOUND IN PATH')
            self.load()

        except Exception as e:

            if 'No such file or directory' in str(e):
                print('\nNO OBJECT FOUND IN PATH')
                print('Creating new one \n')
            else:
                print(e)
                exit()

            # Folder structure
            self.folders = {
            '0_OBR'         : './0_OBR',
            '1_PROCESSED'   : './1_PROCESSED',
            '2_SLICES'      : './2_SLICES',
            '3_DATASET'     : './3_DATASET',
            '4_INFORMATION' : './4_INFORMATION'}

            for key,val in self.folders.items():
                if not os.path.exists(os.path.join(self.path,val)):
                    os.makedirs(os.path.join(self.path,val))

            # Information filenames
            self.INFO = {
            'obr book filename'     :   'obr_book.csv',
            'conditions filename'   :   'conditions.csv',
            'slices book filename'  :   'slices_book.csv',
            'slices filename'       :   'slices.pkl',
            'dataset book filename' :   'dataset_book.csv',
            'dataset filename'      :   'dataset.pkl'}

            # OBR files as an object
            self.obrfiles = dict()

    def genCONDITIONStemplate(self):
        """ Generates a template to set conditions of the OBR readings such as
        relevant interval and characteristics of the beam (for flexural test) """

        if os.path.exists(os.path.join(self.path,self.folders['4_INFORMATION'],self.INFO['conditions filename'])):
            name = self.INFO['conditions filename'].replace('.csv','_template.csv')
        else:
            name = self.INFO['conditions filename']

        print('\n','Creating a conditions file template in:')
        print(os.path.join(self.folders['4_INFORMATION'],name))
        print('please open and edit it')

        # Create template and save it
        df = pd.DataFrame({
                'limit1\n[m]'       : [0],
                'limit2\n[m]'       : [-1],
                'L\n[mm]'           : [1],
                't\n[mm]'           : [1],
                'alpha\n[µm/(m·K)]' : [0]})
        df.to_csv(os.path.join(self.path,self.folders['4_INFORMATION'],name), index=False)

        # Create a .txt to explain
        info =  'limit1: \t stands for the first position to keep [m] \n'
        info += 'limit2: \t stands for the last position to keep [m] \n'
        info += 'L: \t is the lenght of the beam where the fiber is attached in order to perform a flexural test on [mm] \n'
        info += 't: \t is the thickness of the beam [mm] \n'
        info += 'alpha \t stands for CTE [µm/(m·K)]'
        with open(os.path.join(self.path,self.folders['4_INFORMATION'],name.replace('.csv','.txt')), 'w') as f:
            f.write(info)

    def compute(self):
        self.obr()
        self.save()

        self.obr2slices()
        self.save()

        self.slices2dataset()
        self.save()


    from .obr import obr, computeOBR, genOBRbook, obr_ss

    from .obr2slices import obr2slices, gen_slices, genSLICESbook

    from .slices2dataset import slices2dataset, genDATASETbook

    from .save import save

    from .load import load, load_slices, load_dataset

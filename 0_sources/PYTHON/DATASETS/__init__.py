import os
import sys
import pandas as pd


class DATASETS(object):
    """
        Dataset class
        for correct treatment of .obr files

        * See "compute" function below for details
    """

    def __init__(self,path=None,showpath=False):

        # Launch GUI if no path is provided
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

        # In construction generates absolute path and name based on the folder name
        self.path = os.path.abspath(path)
        self.name = f'{os.path.basename(os.path.normpath(path))}.pkl'

        # Just to chek it
        if showpath:
             print(os.listdir(self.path))

        # Tries to load dataset object, else, if not found, creates one
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

            # Creates folder structure if not exists
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


    def compute(self):


        # First read OBR information from . obr files and its filenames
        # only the segment specified in conditions csv will be storaged
        #
        #    takes       : obr files       (storaged in its folder)
        #    generates   : obrfiles object (storaged in self)        and obr book
        #

        self.obr()
        self.save()

        # Then slice each obrfile in obrfiles into slices which will be identified
        # with position, relative position, deflection and temperature among others
        #
        #    takes       : obrfiles object (storaged in self)        and obr    book
        #    generates   : slices   object (storaged in its folder)  and slices book
        #

        self.obr2slices()
        self.save()


        # Finally each two slices, which match in position, input data for NN
        # will be generated by preprocessing its signals ("layer0" or "layer00")
        # whereas output data for NN will be generated out of
        # position, relative deflection and temperature increment
        #
        #    takes       : slices  object (storaged in its folder) and slices  book
        #    generates   : dataset object (storaged in its folder) and dataset book
        #

        self.slices2dataset()
        self.save()

    from .conditions import genCONDITIONStemplate

    from .obr import obr, computeOBR, genOBRbook, obr_ss

    from .obr2slices import obr2slices, gen_slices, genSLICESbook

    from .slices2dataset import slices2dataset, genDATASETbook

    from .save import save

    from .load import load, load_slices, load_dataset

    from .clear import clear_slices, clear_dataset

    from .plots import dataset_plot

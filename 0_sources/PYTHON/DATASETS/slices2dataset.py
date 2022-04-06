import numpy as np
import pandas as pd
import sys
import os
from .obr2slices import slices
from .zero_layers import layer0 as layer0

sys.path.append(os.path.join(os.path.dirname(__file__), '../0_sources/PYTHON'))
from utils import printProgressBar

import numpy as np


class dataset(object):
    """ Class to contain the dataset which will be used to train deep learning models """

    def __init__(self,path,name):

        self.path   = path
        self.name   = name

        try:
            self.load()
            print('\nDATASET found!')

        except Exception as e:

            if 'No such file or directory' in str(e):
                print('\nNO DATASET FOUND IN PATH')
                print('Creating new one \n')
            else:
                print(e)
                exit()

            self.X = list()
            self.Y = list()
            self.parents = list()

    from .save import save
    from .load import load


def slices2dataset(self,matches = 100,percentage=100,avoid_segment=[None, None],conserve_segment=[None,None],more_info=True):
    """ Function to load all slices (previously generated), compute them in pairs, and
    genenerate new values for a dataset

        optional: matches (float)                   : percentage of reference segments to consider from total
        optional: avoid segement (list of floats)   : interval to avoid, in meters [INI, FIN]

        returns: dataset_obj (dataset object)       : object which contains all the dataset information

    """

    # Paths

    slices_path         = os.path.join(self.path,self.folders['2_SLICES'],self.INFO['slices filename'])
    dataset_path        = os.path.join(self.path,self.folders['3_DATASET'],self.INFO['dataset filename'])
    slices_book_path    = os.path.join(self.path,self.folders['4_INFORMATION'],self.INFO['slices book filename'])
    dataset_book_path   = os.path.join(self.path,self.folders['4_INFORMATION'],self.INFO['dataset book filename'])
    conditions_file     = os.path.join(self.path,self.folders['4_INFORMATION'],self.INFO['conditions filename'])

    """ Conditions checkout """

    # Conditions file checkout
    if not os.path.isfile(conditions_file):
        print('\nNo conditions file found')
        self.genCONDITIONStemplate()
        return

    conditions_df = pd.read_csv(conditions_file)
    L = float(conditions_df['L\n[mm]'][0])  * 1e-3               # mm to m
    t = float(conditions_df['t\n[mm]'][0])  * 1e-3               # mm to m
    alpha = float(conditions_df['alpha\n[µm/(m·K)]'])            # already microdeformations


    """ Slices checkout """

    # Check if slices are already created

    if not os.path.exists(slices_path) or not os.path.exists(slices_book_path):
        print('\n','No SLICES created ')
        print('Please run DATASET.obr2slices() to create it','\n')
        return
    else:
        pass

    # Check if slices book already exists

    if not os.path.exists(slices_path) or not os.path.exists(slices_book_path):
        print('\n','No SLICES BOOK created ')
        print('Please run DATASET.genSLICESbook() to create it','\n')
        return
    else:
        pass

    # Load slices
    slices_obj  = slices(os.path.join(self.path,self.folders['2_SLICES']),self.INFO['slices filename'])
    slices_book = pd.read_csv(slices_book_path)

    # Reduce slices
    slices_book = slices_book.sample(frac=percentage/100)

    """ Dataset checkout """

    # Check if dataset already exists
    if os.path.exists(dataset_path):
        ans = input('\nDATASET already computed (append/overwrite/quit):')
        if 'a' in ans:
            pass
        if 'o' in ans:
            self.clear_dataset(auto=True)
        if 'q' in ans:
            return

    # Check if dataset book already exists
    column_names = ['delta_T','delta_EPS','parent1','parent2','spectralshift','x','d_flecha']
    if not os.path.isfile(dataset_book_path):
        df = pd.DataFrame(columns = column_names,dtype=object)
        df.to_csv(dataset_book_path,index=False)


    """ Dataset generation """

    # Create a dataframe to storage new information
    new_information = pd.DataFrame(columns = column_names,dtype=object)

    # Load/initialize dataset
    dataset_obj = dataset(os.path.join(self.path,self.folders['3_DATASET']),self.INFO['dataset filename'])

    # Generate dataset
    LEN = int(len(slices_book.index)); i=0; elements=0
    printProgressBar(0, LEN, prefix = 'Progress:', suffix = 'Complete', length = 50); i += 1

    for index, row in slices_book.iterrows():

        # Avoid certain lenght (noise or whatever)
        if (isinstance(avoid_segment[0], float) or isinstance(avoid_segment[0], int))  and (isinstance(avoid_segment[1], float) or isinstance(avoid_segment[1], int)):
            if avoid_segment[0] <= float(row['x']) and float(row['x']) <= avoid_segment[1]:
                continue
        if (isinstance(conserve_segment[0], float) or isinstance(conserve_segment[0], int))  and (isinstance(conserve_segment[1], float) or isinstance(conserve_segment[1], int)):
            if float(row['x']) <= conserve_segment[0]  and conserve_segment[1] <= float(row['x']):
                continue

        # Search in dataframe for other rows to consider as reference
        try:
            ref_rows = slices_book[(round(slices_book['z'], 8) == round(row['z'],8))]
        except:
            print('NO REF FOUND for:',row['ID'])
            z_matches = slices_book[slices_book['z'] == row['z']]
            print('z_matches: ',len(z_matches))
            continue

        # Reduce number of references
        ref_rows = ref_rows.sample(frac=matches/100)

        # For each slice in the same position compute difference
        for jndex, ref_row in ref_rows.iterrows():

            # Create frecuency array
            f = np.linspace(row['f_0'],row['f_end'],3)

            # Access signal data from slices, preprocess it and then save in dataset as input for NN
            data        = [[0],[0]]
            ref_data    = [[0],[0]]

            data[0]     = slices_obj.slices[row['ID']].P
            data[1]     = slices_obj.slices[row['ID']].S

            ref_data[0] = slices_obj.slices[ref_row['ID']].P
            ref_data[1] = slices_obj.slices[ref_row['ID']].S

            X = layer0(data,ref_data,f)
            dataset_obj.X.append(X)

            # Access status information of the slice to create outputs for NN

            delta_T         = row['temperature']-ref_row['temperature']     # K
            delta_flecha    = (row['flecha']-ref_row['flecha']) * 1e-3      # mm to m
            x               = row['x'] * 1e-3                               # mm to m

            eps_mec = 3*delta_flecha*t/(2*L**3) * (x-L) * 1e6               # Mechanical microdeformations
            eps_the = alpha * delta_T                                       # Thermal  microdeformations
            delta_EPS = eps_mec + eps_the                                   # Total microdeformations

            dataset_obj.Y.append(np.array([delta_T,delta_EPS]))

            # Take information of which slices compounds each dataset

            parent1 = row['ID']
            parent2 = ref_row['ID']

            dataset_obj.parents.append([parent1,parent2])

            # Information to book

            if more_info:
                new_row = {
                'delta_T'   : delta_T,
                'delta_EPS' : delta_EPS,
                'parent1'   : parent1,
                'parent2'   : parent2,
                'spectralshift' : X[0],
                'x'         : row['x'],
                'd_flecha'  : delta_flecha}
            else:
                new_row = {
                'delta_T'   : delta_T,
                'delta_EPS' : delta_EPS,
                'parent1'   : parent1,
                'parent2'   : parent2}

            new_information = new_information.append(new_row, ignore_index=True)

            elements += 1

        printProgressBar(i, LEN, prefix = 'Progress:', suffix = 'Complete', length = 50); i += 1

    """ Save information """

    # Update book with new information
    print(f'Dataset with {elements} elements created!')
    new_information.to_csv(dataset_book_path, mode='a', header=False,index=False)

    # Save/Update dataset
    dataset_obj.save()

    return dataset_obj

def genDATASETbook(self):
    print('Under construction')

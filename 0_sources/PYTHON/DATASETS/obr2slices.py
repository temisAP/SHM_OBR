import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from UTILS.utils import printProgressBar

class a_slice(object):
    """
    Container class for one slice
    """

    def __init__(slice_obj):
        slice_obj.ID           = 0
        slice_obj.temperature  = 0
        slice_obj.flecha       = 0
        slice_obj.z            = 0
        slice_obj.x            = 0
        slice_obj.z_0          = 0
        slice_obj.z_end        = 0
        slice_obj.f_0          = 0
        slice_obj.f_end        = 0
        slice_obj.delta        = 0
        slice_obj.window       = 0
        slice_obj.date         = ''
        slice_obj.parent_file  = ''
        slice_obj.P            = list()
        slice_obj.S            = list()

class slices(object):
    """
    Class to contain a dataset of single slices
    """

    def __init__(self,path,name):

        self.path   = path
        self.name   = name

        try:
            self.load()
            print('\nSLICES found!')

        except Exception as e:

            if 'No such file or directory' in str(e):
                print('\nNO SLICES FOUND IN PATH')
                print('Creating new ones \n')
            else:
                print(e)
                exit()

            self.last_ID = -1
            self.slices  = dict()

    from .save import save
    from .load import load


def obr2slices(self,delta=2000,window=1000):

    """
    Creates a slices object full of slices by slicing .obr data contained in self.obrfiles

            optional: delta  = 2000     : step
            optional: window = 1000     : window

    """

    """ OBR checkout """

    # Check if obr files are already computed
    if len(self.obrfiles) == 0:
        print('\n', 'No obr book created, creating and computing ...','\n')
        self.obr()

    if not any([hasattr(obrfile, 'Data') for key, obrfile in self.obrfiles.items()]):
        print('\n','No data in obr files, computing...','\n')
        self.computeOBR()
    else:
        print('\n','OBR data already computed','\n')
        pass

    """ Slices checkout """

    # Check if slices were previously created
    if os.path.exists(os.path.join(self.path,self.folders['2_SLICES'],self.INFO['slices filename'])):
        ans = input('\nSLICES already computed (append/overwrite/quit):')
        if 'a' in ans:
            pass
        if 'o' in ans:
            self.clear_slices(auto=True)
        if 'q' in ans:
            return

    """ Slices generation """

    # Open/create slices object
    slices_obj = slices(os.path.join(self.path,self.folders['2_SLICES']),self.INFO['slices filename'])

    # Generate slices
    LEN = len(self.obrfiles);i=0
    printProgressBar(0, LEN, prefix = 'Progress:', suffix = 'Complete', length = 50); i += 1

    for key, obrfile in self.obrfiles.items():
        slices_obj = self.gen_slices(obrfile,slices_obj,delta=delta,window=window)
        printProgressBar(i, LEN, prefix = 'Progress:', suffix = 'Complete', length = 50);i += 1

    # Save updated slices object
    slices_obj.save()

def gen_slices(self,obrfile,slices_obj,delta=2000,window=1000):
    """
    Generates slices from an OBR and labels the slices in slices book.

    The slices are created by taking the entire signal and then
    traversing it in jumps of points of size "step" and considering
    the points to be in a bubble of radius "window".

            param: obrfile    (obrfile object)    : an object which contains all the information required
            param: slices_obj (object)            : the slices object which will contain the slices and that will be saved

            optional: delta  = 2000     : step
            optional: window = 1000     : half window size

            returns: slices_obj (slices object): an object which contains all the slices
    """

    # Paths
    book_path   =       os.path.join(self.path,self.folders['4_INFORMATION'],self.INFO['slices book filename'])
    conditions_file =   os.path.join(self.path,self.folders['4_INFORMATION'],self.INFO['conditions filename'])

    """ Checkout and initialization """

    # Conditions file checkout
    if not os.path.exists(conditions_file):
        print('\nNo conditions file found')
        self.genCONDITIONStemplate()
        exit()

    # Column names and slice initialization
    a_slice_obj = a_slice()
    column_names = [key for key in a_slice_obj.__dict__.keys()]

    # Check if book exists
    if not os.path.exists(book_path):
        df = pd.DataFrame(columns = column_names, dtype=object)
        df.to_csv(book_path,index=False)

    # Create a dataframe to storage new information
    new_information = pd.DataFrame(columns = column_names, dtype=object)

    """ Get information about state of the fiber """

    # Full data length
    n = len(obrfile.Data[0])

    # Corresponding steps
    steps = range(window,n-window+1,delta)

    # Get beam characteristics and status of the beam
    temperature = obrfile.temperature
    flecha      = obrfile.flecha

    # Relative position on the beam
    conditions_df = pd.read_csv(conditions_file)
    L = float(conditions_df['L\n[mm]'][0])
    x = np.linspace(0,L,len(steps))


    """ Slices generation """

    # ID initialization
    ID = slices_obj.last_ID

    # Generate slices
    for idx,i in enumerate(steps):

            # Current ID
            ID += 1

            # Lenght
            z = obrfile.z[i-window:i+window]

            # Update new information
            new_row = {
            'ID'                : ID,
            'temperature'       : temperature,          # ºC
            'flecha'            : flecha,               # mm
            'z'                 : obrfile.z[i],         # m
            'x'                 : x[idx],               # mm
            'z_0'               : z[0] * 1e3,           # mm
            'z_end'             : z[-1]* 1e3,           # mm
            'f_0'               : obrfile.f[0],         # GHz
            'f_end'             : obrfile.f[-1],        # GHz
            'delta'             : delta,
            'window'            : window,
            'date'              : time.strftime("%Y,%m,%d,%H:%M:%S"),
            'parent_file'       : obrfile.name,
            'P'                 : obrfile.Data[0][i-window:i+window],
            'S'                 : obrfile.Data[1][i-window:i+window]}

            # Append new row
            new_information = new_information.append(new_row, ignore_index=True)

            # Append new element
            a_slice_obj = a_slice()
            for val in column_names:
                setattr(a_slice_obj, val, new_row[val])
            slices_obj.slices[ID] = a_slice_obj

    # Update last ID
    slices_obj.last_ID = ID

    # Append new_information to book_path but without 'P' and 'S' fields (too much data)
    df = pd.DataFrame(new_information)
    df = df.drop(['P','S'], axis=1)
    df.to_csv(book_path, mode='a', header=False,index=False)

    return slices_obj


def genSLICESbook(self):
    print('Under construction')

import numpy as np
import pandas as pd
import sys
import os
from .obr2slices import slices

sys.path.append(os.path.join(os.path.dirname(__file__), '../0_sources/PYTHON'))
from utils import printProgressBar

import numpy as np
import statsmodels.api as sm

class dataset(object):
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

def layer0(data,ref_data,f,lamb=25, mode='same'):
    """ Zero 'layer' for a IA model, it takes signal data and its reference
    and computes autocorrelation comparisson and cross correlation

        param: data (2xn np.array)      : array which contains the signal data
        param: ref_data (2xn np.array)  : array which contains the reference signal data
        param: f (1D np.array)          : frequency sampling array

            *Both data and ref_data are compound with the same structure:
                    data = [[P],[S]]

        optional: lamb=25       : lambda parameter for the STL filter performed over
                                  autocorrelation comparisson
        optional: mode='same'   : mode for np.correlate function

        returns: X (1D np.array): array which contains cross and auto autocorrelation
                                  information: [[spectralshift],crosscorr,autocorr]

    """

    P1 = data[0]
    S1 = data[1]
    P2 = ref_data[0]
    S2 = ref_data[1]

    """ Autocorrelation """

    # Autocorrelation
    y = P1
    autocorr1 = np.correlate(y, y, mode=mode)/np.var(y)/len(y)
    y = P2
    autocorr2 = np.correlate(y, y, mode=mode)/np.var(y)/len(y)
    autocorr = np.absolute(autocorr1-autocorr2)

    # STL filter
    seasonal,trend = np.array(sm.tsa.filters.hpfilter(autocorr, lamb=25))
    autocorr = trend

    autocorr = autocorr[int(len(autocorr)/2-200):int(len(autocorr)/2+200)]

    """ Cross correlation """

    y1 = P1
    y2 = P2

    # Frequency sampling
    DF = f[-1]-f[0]     # Frequency increment
    n = len(y1)         # Sample lenght
    sr = 1/(DF/n)       # Scan ratio

    # FFT
    Y1 = np.absolute(np.fft.fft(y1))
    Y2 = np.absolute(np.fft.fft(y2))
    Y1 = (Y1 - np.mean(Y1)) / (np.std(Y1) * len(Y1))
    Y2 = (Y2 - np.mean(Y2)) / (np.std(Y2))

    # Cross corelation
    crosscorr = np.correlate(Y1, Y2, mode=mode)

    # Spectral shift
    spectralshift_lags = np.linspace(-0.5*n/sr, 0.5*n/sr, n+1)
    spectralshift = spectralshift_lags[np.argmax(crosscorr)]
    spectralshift = -1*spectralshift/np.mean(f)*1e6             # micro spectralshift

    """ Return """

    t = [[spectralshift],crosscorr,autocorr]
    X = np.array([item for sublist in t for item in sublist])

    return X

def layer00(data,ref_data,f):

    P1 = data[0]
    S1 = data[1]
    P2 = ref_data[0]
    S2 = ref_data[1]

    P1 = np.absolute(np.fft.fft(P1))
    P2 = np.absolute(np.fft.fft(P2))
    P1 = (P1 - np.mean(P1)) / (np.std(P1) * len(P1))
    P2 = (P2 - np.mean(P2)) / (np.std(P2))

    S1 = np.absolute(np.fft.fft(S1))
    S2 = np.absolute(np.fft.fft(S2))
    S1 = (S1 - np.mean(S1)) / (np.std(S1) * len(S1))
    S2 = (S2 - np.mean(S2)) / (np.std(S2))

    t = [P1.real, P1.imag,
        S1.real, S1.imag,
        P2.real, P2.imag,
        S2.real, S2.imag]

    X = np.array([item for sublist in t for item in sublist])

    return X

def slices2dataset(self,matches = 100,percentage=100,avoid_segment=[None, None]):
    """ Function to load all slices (previously generated), compute them in pairs, and
    genenerate new values for a dataset

        optional: matches (float)                   : percentage of reference segments to consider from total
        optional: avoid segement (list of floats)   : interval to avoid, in meters [INI, FIN]
    """

    # Paths

    slices_path         = os.path.join(self.path,self.folders['2_SLICES'],self.INFO['slices filename'])
    dataset_path        = os.path.join(self.path,self.folders['3_DATASET'],self.INFO['dataset filename'])
    slices_book_path    = os.path.join(self.path,self.folders['4_INFORMATION'],self.INFO['slices book filename'])
    dataset_book_path   = os.path.join(self.path,self.folders['4_INFORMATION'],self.INFO['dataset book filename'])
    conditions_file =   os.path.join(self.path,self.folders['4_INFORMATION'],self.INFO['conditions filename'])

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
        print('\n','DATASET already computed')
        if 'n' in input('Do you want to append new content? (yes/no)'):
            return
        else:
            pass

    # Check if dataset book already exists
    column_names = ['delta_T','delta_EPS','parent1','parent2','spectralshift']
    if not os.path.isfile(dataset_book_path):
        df = pd.DataFrame(columns = column_names,dtype=object)
        df.to_csv(dataset_book_path,index=False)

    # Create a dataframe to storage new information
    new_information = pd.DataFrame(columns = column_names,dtype=object)

    # Load/initialize dataset
    dataset_obj = dataset(os.path.join(self.path,self.folders['3_DATASET']),self.INFO['dataset filename'])

    """ Dataset generation """

    # Initialize lists of lists
    data     = [[0],[0]]
    ref_data = [[0],[0]]

    # Generate dataset
    LEN = int(len(slices_book.index)); i=0; elements=0
    printProgressBar(0, LEN, prefix = 'Progress:', suffix = 'Complete', length = 50); i += 1

    for index, row in slices_book.iterrows():


        # Avoid certain lenght (noise or whatever)
        if isinstance(avoid_segment[0], float) and isinstance(avoid_segment[1], float):
            if avoid_segment[0] <= slices_book['x'] and slices_book['x'] <= avoid_segment[0]:
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

            # Access data
            data[0]     = slices_obj.slices[row['ID']].P
            data[1]     = slices_obj.slices[row['ID']].S

            ref_data[0] = slices_obj.slices[ref_row['ID']].P
            ref_data[1] = slices_obj.slices[ref_row['ID']].S

            X = layer00(data,ref_data,f)
            dataset_obj.X.append(X)

            delta_T         = row['temperature']-ref_row['temperature']     # K
            delta_flecha    = (row['flecha']-ref_row['flecha']) * 1e-3      # mm to m
            x               = row['x'] * 1e-3                               # mm to m

            eps_mec = 3*delta_flecha*t/(2*L**3) * (x-L) * 1e6               # Mechanical microdeformations
            eps_the = alpha * delta_T                                       # Thermal  microdeformations
            delta_EPS = eps_mec + eps_the                                   # Total microdeformations

            dataset_obj.Y.append(np.array([delta_T,delta_EPS]))

            parent1 = row['ID']
            parent2 = ref_row['ID']

            dataset_obj.parents.append([parent1,parent2])

            # Information to book

            more_info = True
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

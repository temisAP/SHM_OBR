import os
import pandas as pd
import numpy as np
import glob
import re
import sys
import time
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from UTILS.read_obr import multi_read_obr


class obrfile(object):
    """ Container class of obr files """

    def __init__(obrfile_obj,filename,temperature,flecha,date):

        obrfile_obj.filename       = filename
        obrfile_obj.name           = filename.replace('.obr','')
        obrfile_obj.temperature    = temperature
        obrfile_obj.flecha         = flecha
        obrfile_obj.date           = date

def obr(self):

    # First generates obr book from obr filenames and the date recorded in each .obr
    self.genOBRbook()
    # Then open each one and reads relevant information in the specified segment
    self.computeOBR()

def genOBRbook(self):
    """
    Function to generate obr book from obr filenames and the date recorded in each .obr

        returns: dataframe from obr book file

    """

    # Check current information

    book_path = os.path.join(self.path,self.folders['4_INFORMATION'],self.INFO['obr book filename'])

    if not os.path.exists(book_path):
        print('')
        print('OBR book not found')
        print('Creating a new one ...')
        save_df  = True
        save_obj = True
    elif os.path.exists(book_path) and len(self.obrfiles) == 0:
        print('')
        print('OBR book found but not registered')
        print('Registering ...')
        save_df  = False
        save_obj = True
    else:
        print('')
        print('OBR book found and registered')
        if 'n' in input('Do you want to overwrite?(yes/no)'):
            return pd.read_csv(book_path)
        else:
            save_df  = True
            save_obj = True

    # Get all OBR files
    obr_files = find_OBR(self.path)
    # Sort OBR files
    obr_files = sort_OBR(obr_files)
    # Initialize dataframe
    df = pd.DataFrame()

    # Loop through OBR files
    for f in obr_files:
        # Get date
        date = get_date(os.path.join(self.path,'0_OBR',f))
        # Get temperature and flecha
        temperature, flecha = get_status(f)
        # Append to object obrfile
        if save_obj:
            self.obrfiles[f] = obrfile(f,temperature,flecha,date)
        # Append to dataframe
        df = df.append({
            'filename':     f,
            'temperature':  temperature,
            'flecha':       flecha,
            'date':         date},
             ignore_index=True)

    # Save dataframe to csv
    if save_df:
        df.to_csv(book_path, index=False)
        # Print dataframe
        print('\n',df,'\n')
        print('Done!')
        # Return dataframe
        return df
    else:
        print('Done!')
        return df

def computeOBR(self):
    """
    Reads all .obr files and registers information: f,z and Data = [Pc,Sc,Hc]
    among currently existing (filename, name, flecha, temperature and date)

    * If RAM is not able to allocate enough memory the object will be saved and
    by running this function a couple of times all the information will be
    sotoraged correctly

    """

    # Check if information files exists
    if not os.path.exists(os.path.join(self.path,self.folders['4_INFORMATION'],self.INFO['obr book filename'])) and len(self.obrfiles) == 0:
        print('\n','OBR book not found or not registered')
        print('Please, create/register a new one calling')
        print('DATASET.genOBRbook()')
        return
    else:
        pass

    if not os.path.exists(os.path.join(self.path,self.folders['4_INFORMATION'],self.INFO['conditions filename'])):
        print('\n','No conditions found')
        print('Please if you want to crop the data or specify the beam where the fiber was glued run:')
        print('DATASET.genCONDITIONStemplate()')
        print('and edit it')
        return
    else:
        # Gets limits from conditions file
        df = pd.read_csv(os.path.join(self.path,self.folders['4_INFORMATION'],self.INFO['conditions filename']))

        limit1 = float(df['limit1\n[m]'][0])
        limit2 = float(df['limit2\n[m]'][0])

    # Generate datasets from selected data

    for key, obrfile in self.obrfiles.items():

        import psutil
        if psutil.virtual_memory()[2] < 90:

            if not hasattr(obrfile, 'Data'):
                # Read .obr file
                f,z,Data = multi_read_obr([obrfile.name],os.path.join(self.path,self.folders['0_OBR']),limit1=limit1,limit2=limit2)
                # Update OBR file register
                obrfile.f           = f
                obrfile.z           = z
                obrfile.Data        = Data[obrfile.name]
            else:
                pass

        else:
            print('\nUnable to allocate more information')
            print("DON'T PANIC the information will be saved")
            print('just run again DATASETS.computeOBR() until no more .obr files are read')
            self.save()
            return False
            exit()

    return True

from .analisis.obr_ss import obr_ss
from .analisis.obr_TE import obr_TE
from .analisis.filters import obr_filters


def local_analysis(self,files,position):
    """ Function to analize signal in a certain point """

    from .analisis.local_analisis import analysis_10

    analysis_10(self,files,position)

def find_OBR(path):
    """ Function to find all .obr files from a folder

        param:  path      (string)          : path to folder
        return: obr_files (list of string)  : list of OBR filenames

    """
    # Find all .obr files
    obr_files = glob.glob(os.path.join(path,'0_OBR','*.obr'))
    # Keep just filename and extension
    obr_files = [os.path.basename(f) for f in obr_files]
    return obr_files

def sort_OBR(obr_files):
    """
    Funtion to sort OBR by the first number (before '_' splitter)

        param: obr_files (list of string) : list of OBR filenames
        return: obr_files (list of string) : list of OBR filenames sorted

    """
    obr_files.sort(key=lambda x: int(re.findall('\d+', x)[0]))
    return obr_files

def get_status(filename):

    """
    Checks filename format, then extracts temperature and delfection from the name

        Valid formats are:
                [temperature]_grados.obr               -> For just-temperature samples
                [flecha]_mm.obr                        -> For just-flexion samples
                [flecha]_mm_[temperature]_grados.obr   -> For flexion-temperature samples

        returns: Temperature, flecha :temperature and deflection (in mm)

     """

    # Determine type of sample
    if 'grados' in filename and not '_mm_' in filename:
        flecha = 0
        temperature = filename.split('_')[0]
    elif 'mm' in filename and not 'grados' in filename:
        flecha = filename.split('_')[0]
        temperature = 0
    elif 'mm' in filename and 'grados' in filename:
        flecha = filename.split('_')[0]
        temperature = filename.split('_')[2]
    else:
        print('ERROR: File format not recognized')
        return '?','?'

    # Convert temperature to float
    temperature = float(temperature)
    # Convert flecha to float
    flecha = float(flecha)

    return temperature, flecha

def get_date(file):
    """
    Open an .obr file to get date of the measure

        param: file (str): file to be read
        return: DateTime (str): date formated as %Y,%M,%D,%h:%m:%s

    """

    # Lectura de datos
    offset = np.dtype('<f').itemsize
    offset += np.dtype('|U8').itemsize
    offset = 12 # Ni idea de por qué este offset pero funciona
    offset += np.dtype('<d').itemsize
    offset += np.dtype('<d').itemsize
    offset += np.dtype('<d').itemsize
    offset += np.dtype('<d').itemsize
    offset += np.dtype('uint16').itemsize
    offset += np.dtype('<d').itemsize
    offset += np.dtype('int32').itemsize
    offset += np.dtype('int32').itemsize
    offset += np.dtype('uint32').itemsize
    offset += np.dtype('uint32').itemsize

    DateTime=np.fromfile(file, count=8,dtype= 'uint16',offset = offset)                              # Fecha de la medida

    DateTime=f'{DateTime[0]},{DateTime[1]},{DateTime[3]},{DateTime[4]}:{DateTime[5]}:{DateTime[6]}'  # "2022,03,03,13:41:27"

    return DateTime

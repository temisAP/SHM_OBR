import os
import pandas as pd
import numpy as np
import glob
import re
import sys
import time
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from UTILS.utils import find_index
from UTILS.read_obr import multi_read_obr
from SIGNAL.Spectral_Shift import global_spectral_shift
from SIGNAL.wavelet import wavelet
from SIGNAL.chirplet import chirplet
from SIGNAL.STFT import stft
from SIGNAL.spectrogram import spectrogram


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

def obr_ss(self,REF,type=None,eps=False):
    """
    Plots spectral shift from obr files

        param: REF (str): file which take as reference

        optional: type = None (str):  if 'flecha' plots a colorbar and sorts lines by its deflection
                                      if 'temperature' plots a colorbar and sorts lines by its temperature

        optional: eps = False (bool): if True then adds the deformation along the segment in the plot


    """

    """ Conditions checkout """

    # Conditions file checkout
    conditions_file     = os.path.join(self.path,self.folders['4_INFORMATION'],self.INFO['conditions filename'])
    if not os.path.isfile(conditions_file):
        print('\nNo conditions file found')
        self.genCONDITIONStemplate()
        return

    conditions_df = pd.read_csv(conditions_file)
    L = float(conditions_df['L\n[mm]'][0])  * 1e-3               # mm to m
    t = float(conditions_df['t\n[mm]'][0])  * 1e-3               # mm to m
    alpha = float(conditions_df['alpha\n[µm/(m·K)]'])            # already microdeformations

    """ Get filenames to iterate """

    files = list(self.obrfiles.keys())
    Ts = list()
    files.remove(REF)
    f = np.linspace(self.obrfiles[REF].f[0],self.obrfiles[REF].f[-1],3)


    """ Get labels """

    for file in files:
        if hasattr(self.obrfiles[file], 'Data'):

            if type == 'temperature':
                Ts.append(self.obrfiles[file].temperature-self.obrfiles[REF].temperature)
            elif type == 'flecha':
                Ts.append(self.obrfiles[file].flecha-self.obrfiles[REF].flecha)

    """ Get spectral shift from each file and plots it """

    fig, ax = plt.subplots()

    if eps:
        ax2 = ax.twinx()
        ax2.set_ylabel(r'$\Delta \mu\varepsilon$'+'\n'+'(dashed)').set_rotation(0)

    for file in files:
        if hasattr(self.obrfiles[file], 'Data'):

            y1 = self.obrfiles[REF].Data[0]
            y2 = self.obrfiles[file].Data[0]
            z = self.obrfiles[file].z

            if type == 'temperature':
                T = int(self.obrfiles[file].temperature-self.obrfiles[REF].temperature)
            elif type == 'flecha':
                T = int(self.obrfiles[file].flecha-self.obrfiles[REF].flecha)

            spectralshift = global_spectral_shift(y1,y2,f,delta=2000,window=1000,display = False)
            z = np.linspace(z[0],z[-1],len(spectralshift))

            # Add real deformation to graph
            if eps:
                # Get status of the beam
                delta_flecha = (self.obrfiles[file].flecha-self.obrfiles[REF].flecha) * 1e-3             # mm to m
                delta_T      = (self.obrfiles[file].temperature-self.obrfiles[REF].temperature)          # K
                # Relative position on the beam
                x = np.linspace(0,L,len(z))     # in m

                eps_mec = 3*delta_flecha*t/(2*L**3) * (x-L) * 1e6               # Mechanical microdeformations
                eps_the = alpha * delta_T * np.ones_like(x)                     # Thermal  microdeformations
                delta_EPS = eps_mec + eps_the                                   # Total microdeformations

                ax2.plot(z,delta_EPS,'--',label=file,color=plt.cm.jet(find_index(Ts,T)/len(Ts)))

            if type == None:
                ax.plot(z,spectralshift,label=file)
            else:
                ax.plot(z,spectralshift,label=file,color=plt.cm.jet(find_index(Ts,T)/len(Ts)))

    # Legend or colorbar
    if type == None:
        ax.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    else:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=min(Ts), vmax=max(Ts)))
        cbar = plt.colorbar(sm,ax=ax,spacing='proportional')
        if type == 'temperature':
            cbar.set_label(r'$\Delta T$ [K]',rotation=0,labelpad=15)
        elif type == 'flecha':
            cbar.set_label(r'$\delta$ [mm]',rotation=0,labelpad=15)


    ax.set_xlabel('z [m]')
    ax.set_ylabel(r'$\frac{-\Delta \nu}{\bar{\nu}}$',fontsize=16).set_rotation(0)
    ax.grid()
    ax2.grid() if eps else False
    plt.show()

def local_analysis(self,files,position):
    """ Function to analize signal in a certain point """

    # Experiments from here

    analysis_1 = False
    analysis_2 = True

    if analysis_1:
        import matplotlib.pyplot as plt
        import scipy.signal as signal
        # Compute chirplets

        REF = files[0]
        chirplets = dict.fromkeys(files)
        for file in files:
            y = self.obrfiles[file].Data[0]
            i = find_index(self.obrfiles[file].z,position)

            window = 1000

            y = y[i-window:i+window]

            chirplets[file] = chirplet(np.abs(y),plot=False)
            #wavelet(y,plot=True)

            plt.title(file)
            plt.close()



        # 2d Correlation of chirplets

        for file in files:


            corr = signal.correlate2d(chirplets[REF], chirplets[file], boundary='wrap', mode='full') # mode full es clave (la diferencia es más marcada) , fill bounds hace que sea redondo, wrap hace bandas y symmetry hace que se pierda la simetría xd
            y, x = np.unravel_index(np.argmax(corr), corr.shape)  # find the match

            import matplotlib.pyplot as plt

            v_max = 6e-5 # np.amax(corr)
            v_min = 1e-5 # np.amin(corr)

            fig, (ax_orig, ax_template, ax_corr) = plt.subplots(3, 1)

            ax_orig.contour(chirplets[REF], cmap='jet')
            sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=np.amin(chirplets[REF]), vmax=np.amax(chirplets[REF])))
            cbar = plt.colorbar(sm,ax=ax_orig,spacing='proportional')
            ax_orig.set_title(REF)
            ax_orig.grid()

            ax_template.contour(chirplets[file], cmap='jet')
            sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=np.amin(chirplets[file]), vmax=np.amax(chirplets[file])))
            cbar = plt.colorbar(sm,ax=ax_template,spacing='proportional')
            ax_template.set_title(file)
            ax_template.grid()


            ax_corr.contour(corr, cmap='jet')
            sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=v_min, vmax=v_max))
            cbar = plt.colorbar(sm,ax=ax_corr,spacing='proportional')
            ax_corr.set_title('Cross-correlation')
            ax_corr.plot(x, y, 'ko')
            ax_corr.grid()

            print(x,y)

            fig.tight_layout()


        plt.show()


        if False:
            import csv
            with open(os.path.join(self.path,self.folders['1_PROCESSED'],'data1.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(y.real)
                writer.writerow(y.imag)

    if analysis_2:

        import matplotlib.pyplot as plt
        import scipy.signal as signal

        v_max = 6e-5
        v_min = 1e-5

        delta = 0.05
        positions = [position-delta,position,position+delta]
        data = {f'{positions[0]}':dict.fromkeys(files),
                f'{positions[1]}':dict.fromkeys(files),
                f'{positions[2]}':dict.fromkeys(files)}

        for pos in positions:

            # Compute chirplets
            for file in files:
                y = self.obrfiles[file].Data[0]
                i = find_index(self.obrfiles[file].z,pos)

                window = 1000
                data[str(pos)][file] = chirplet(np.abs(y[i-window:i+window]),plot=False)

        # 2d Correlation of chirplets
        REF = files[0]
        fig, ax = plt.subplots(3, 4)

        idx = 0
        for key in data.keys():
            jdx = 0
            for file in files:

                corr = signal.correlate2d(data[key][REF], data[key][file], boundary='fill', mode='same') # mode full es clave (la diferencia es más marcada) , fill bounds hace que sea redondo, wrap hace bandas y symmetry hace que se pierda la simetría xd
                y, x = np.unravel_index(np.argmax(corr), corr.shape)  # find the match

                ax[idx%3,jdx%4].contour(corr, cmap='jet')
                sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=v_min, vmax=v_max))
                cbar = plt.colorbar(sm,ax=ax[idx%3,jdx%4],spacing='proportional')
                ax[idx%3,jdx%4].set_title(f'{file} at {key} m')
                ax[idx%3,jdx%4].plot(x, y, 'ko')
                ax[idx%3,jdx%4].grid()

                print(f'{file} \t at {key}: {x} {y}')

                fig.tight_layout()
                jdx += 1
            idx += 1


        plt.show()


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

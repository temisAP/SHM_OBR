def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    if float(percent) <= 100:
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    else:
        print('OVER-COMPLETED',end = '\n')
    # Print New Line on Complete
    if iteration == total:
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\n')

def find_index(array,value):

    """ Function to determine the closest index to some value
        :param array (list) :   list to be evaluated
        :param value (float):   value to reach
        :param value (array):   values to reach

        :return idx  (int/list)  :   index (or indexes if value is array)
    """

    import numpy as np

    array = np.asarray(array)

    if isinstance(value, int) or isinstance(value, float):
        idx = (np.abs(array - value)).argmin()
        return idx

    elif isinstance(value, list) or type(value).__module__ == np.__name__:
        idx = list()
        for val in value:
            idx.append((np.abs(array - val)).argmin())
        return idx
    else:
        print(f'find_index error [Type not suported: {type(value)}]')
        exit()


def get_all_files(path,verbosity = False,extension=False,file_extension='.obr'):
    """ Get all binaries filenames from a path """

    import os

    binaries = []
    for root, dirs, files in os.walk(path):
        print((len(path) - 1) * '---', os.path.basename(root)) if verbosity == True else False
        for file in files:
            print(len(path) * '---', file) if verbosity == True else False
            if file.endswith(file_extension):
                if extension == False:
                    binaries.append(file.split('.')[0])
                elif extension == True:
                    binaries.append(file)

    try:
        # Split filenames and get the first number
        order = list()
        for file in binaries:
            try:
                num =  float(file.split('_')[0])
                order.append([num , file])
            except:
                try:
                    num =  float(file.split('.')[0])
                    order.append([num , file])
                except:
                    pass
        # Sort files using the num-> file correspondence  in "order"
        if order != []:
            binaries = [file for num,file in sorted(order, key = lambda x:x[0])]
    except:
        pass


    return binaries

def get_times(file):
    """ Function to get elapsed time in (HH:MM:SS) format second column of a csv
    where elapsed time is sepecified in seconds.

        : param file   (string): path to file

        : retrun times (np.array): array with elapsed times
    """

    import numpy as np
    import pandas as pd
    import time

    # Read file
    data = pd.read_csv(file, sep=',', header=None)

    # Convert to numpy array
    data = data.values

    # Get values
    elapsed_time = data[:, 1]
    # Change format
    times = np.array([time.strftime('%H:%M:%S', time.gmtime(t)) for t in elapsed_time])

    return times


def create_onedrive_directdownload(onedrive_link):
    import base64
    data_bytes64 = base64.b64encode(bytes(onedrive_link, 'utf-8'))
    data_bytes64_String = data_bytes64.decode('utf-8').replace('/','_').replace('+','-').rstrip("=")
    resultUrl = f"https://api.onedrive.com/v1.0/shares/u!{data_bytes64_String}/root/content"
    return resultUrl

def find_OBR(path):
    """ Function to find all .obr files from a folder

        param: path (string): path to folder
        return: obr_files (list of string): list of OBR filenames

    """
    # Find all .obr files
    obr_files = glob.glob(os.path.join(path,'0_OBR','*.obr'))
    # Keep just filename and extension
    obr_files = [os.path.basename(f) for f in obr_files]
    return obr_files

def sort_OBR(obr_files):
    """ Funtion to sort OBR by the first number (before '_' splitter)

        param: obr_files (list of string) : list of OBR filenames
        return: obr_files (list of string) : list of OBR filenames sorted

    """
    obr_files.sort(key=lambda x: int(re.findall('\d+', x)[0]))
    return obr_files

def remove_extension(obr_files):

    return [obr_file.replace('.obr','') for obr_file in obr_files]

def find_all_OBR(path):
    obr_files = find_OBR(path)
    obr_files = sort_OBR(obr_files)
    obr_files = remove_extension(obr_files)

    return obr_files

def get_status(filename):

    """ Checks filename format, then extracts temperature and delfection from the name

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

def check_memory(percentage=90,timeout = 60):

    import time
    import psutil

    zero_time = float(time.time())
    while psutil.virtual_memory()[2] > percentage:
        print('Waiting for memory')
        time.sleep(1)
        if float(time.time())-zero_time > timeout:
            exit()

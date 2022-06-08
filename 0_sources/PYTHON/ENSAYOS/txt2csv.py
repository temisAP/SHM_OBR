import os
import sys
import pandas as pd
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

def txt2csv(folder_path):

    # Read all .txt files on the folder path
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    # Filename structure: EX_NN, X is number of segment and NN load
    # Filenames must be sorted first by number of segment and then in increasing load
    files.sort(key=lambda x: (int(x.split('_')[0].replace('E','')), float(x.split('_')[1].replace('.txt',''))))

    # Get all files basenames
    basenames = [os.path.splitext(f)[0] for f in files]

    # Create a pandas dataframe with the following columns: basename, z, deformation, path
    df = pd.DataFrame(columns=['Stage','Load (kN)', 'Length (m)', 'Strain (microstrain)','\t'])

    # Iterate over all files
    for i, f in enumerate(files):

        # Get the file path
        file_path = os.path.join(folder_path, f)

        # Get the basename
        basename = basenames[i]

        # Get the z and strain values
        z = list()
        val = list()
        write = False
        with open(file_path, 'r') as fo:
            for line in fo:
                if 'Length (m)' in line:
                    write = True
                    continue
                if write:
                    z.append(float(line.split('\t')[0]))
                    val.append(float(line.split('\t')[1]))

        # Add the data to the dataframe
        df.loc[i] = [basename.split('_')[0],basename.split('_')[1], z, val,'\t']

    # Save the dataframe to a .csv file
    df.to_csv(os.path.join(folder_path, 'data.csv'), index=False)
    print('Done!')

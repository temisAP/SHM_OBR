import os
import pandas as pd

def genCONDITIONStemplate(self):
    """
    Generates a template to set conditions of the OBR readings such as
    relevant interval and characteristics of the beam (for flexural test)

    """

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
    info += 'L: \t is the lenght of the beam where the fiber is attached in order to perform a flexural test on [mm]\n'
    info += 't: \t is the thickness of the beam [mm] \n'
    info += 'alpha \t stands for CTE [µm/(m·K)] \n'
    info += '\n'
    info += '\n'
    info += 'Set L and y as 1 and alpha as 0 if the fiber is not glued anywhere'

    with open(os.path.join(self.path,self.folders['4_INFORMATION'],name.replace('.csv','.txt')), 'w') as f:
        f.write(info)

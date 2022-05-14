import os
import sys
import numpy as np
import psutil

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from UTILS.sensor import sensor

class measure(object):
    """
        A class to contain preformed measurements along the fiber
    """
    def __init__(self,x,T,E,reference,file):

        self.x = x                # Lenght axis distribution            [mm]
        self.T = T                # Temperature increment distribution  [K]
        self.E = E                # Deformation increment distribution  [με]
        self.reference = reference  # OBR File used as reference
        self.file      = file       # OBR File of the current state

def obr2measures(self,REFs=None,files=None,delta=200,window=1000):

    files = files if files else list(self.obrfiles.keys())
    REFs  = REFs  if REFs  else list(self.obrfiles.keys())

    if not hasattr(self,'measures'):
        self.measures = dict.fromkeys(self.obrfiles.keys())
        for REF in self.obrfiles.keys():
            self.measures[REF] = dict.fromkeys(self.obrfiles.keys())
            for file in self.obrfiles.keys():
                self.measures[REF][file] = None

    if not isinstance(REFs,list):
        REFs = [REFs]

    if not isinstance(files ,list):
        files = [files]


    for REF in REFs:

        refData = self.obrfiles[REF].Data
        z       = self.obrfiles[REF].z
        f       = self.obrfiles[REF].f

        for file in files:

            if psutil.virtual_memory()[2] < 90:

                if not self.measures[REF][file]:
                     Data  = self.obrfiles[file].Data
                     print(f'\nComputing measurement between {REF} and {file}')
                     T,E = sensor(Data,refData,f,delta=delta,window=window,display = False)   # compute measurements
                     x = np.linspace(z[0],z[-1],len(T)) * 1e3                                  # m to mm
                     self.measures[REF][file] = measure(x,T,E,REF,file)

                else:
                    print('\nUnable to allocate more information')
                    print("DON'T PANIC the information will be saved")
                    print('just run again DATASETS.obr2measures() until no more .obr files are read')
                    self.save()
                    return False
                    exit()

    self.save_measures()

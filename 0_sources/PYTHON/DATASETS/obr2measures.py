import os
import sys

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
        self.reference = parent1  # OBR File used as reference
        self.file      = parent2  # OBR File of the current state

def obr2measures(self,REFs=None,files=None,delta=200,window=1000):

    files = files if files else self.obrfiles.keys()
    REFs  = REFs  if REFs  else self.obrfiles.keys()

    if not hasattr(self,'measures'):
        self.measures = dict.fromkeys(self.obrfiles.keys())

    if not isinstance(REFs,list):
        REFs = [REFs]

    if not isinstance(files ,list):
        files = [files]

    for REF in REFs:
        refData = self.obrfiles[REF].Data
        z       = self.obrfiles[REF].z
        for file in files:
             Data  = self.obrfiles[file].Data
             print('\nComputing measurement between {REF} and {file}')
             T,E = sensor(Data,refData,f,delta=delta,window=window,display = False)   # compute measurements
             x = np.linspace(z[0],z[-1],len(TT)) * 1e3                                  # m to mm
             self.measures[REF][file] = measure(x,T,E,REF,file)

    self.save_measures()

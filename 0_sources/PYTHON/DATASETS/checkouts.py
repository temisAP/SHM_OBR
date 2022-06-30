import os

def conditions_checkout(self,stop=True):
    """ Conditions file checkout """

    conditions_file =   os.path.join(self.path,self.folders['4_INFORMATION'],self.INFO['conditions filename'])

    if not os.path.exists(conditions_file):
        print('\nNo conditions file found')
        self.genCONDITIONStemplate()
        exit() if stop else None

def obr_checkout(self):

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

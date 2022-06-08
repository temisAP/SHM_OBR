import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../0_sources/PYTHON'))
from ENSAYOS.txt2csv import txt2csv

folder_path = '/mnt/sda/0_Andres/1_Universidad/SHM/98_data/0_Ensayos/EduardoTorroja_ConcreteBeam/1_PROCESSED'

txt2csv(folder_path)

import sys
import os
from .utils import find_index
from .utils import stokes_vector

import numpy as np
import matplotlib.pyplot as plt

def read_obr(file):
    """
        Function to read binaries (OBR)

        param
            file    (string): path al fichero a leer

        returns
            f       (np.array)  : Distribución espectral [GHz] (de SF a SF+FI*2*n en incrementos de FI)
            z       (np.array)  : Distribución espacial  [m]
            Pc      (complex 1xn np.array)  : Medida de la polarización p
            Sc      (complex 1xn np.array)  : Medida de la polarización s
            St      (4xn np.array)          : Vector de Stokes

        * Along this file some information has been commented out for a faster reading
          please feel free of uncoment whenever you may need

    """

    # Lectura de datos

    #FileForVer=np.fromfile(file, count=1, dtype= np.dtype('<f'))[0]                 # Versi�n del formato del archivo (3.4)
    offset = np.dtype('<f').itemsize

    #ObrOfdr=np.fromfile(file, count=8,dtype= '|S1', offset = offset).astype('|U8')  # Tipo de archivo. String sin sentido
    offset += np.dtype('|U8').itemsize
    offset = 12 # Ni idea de por qué este offset pero funciona

    StartFreq=np.fromfile(file, count=1,dtype= np.dtype('<d'), offset = offset)[0]  # [GHz] Frecuencia de inicio de scan, es la más alta de todo el espectro escaneado
    offset += np.dtype('<d').itemsize

    SF=StartFreq

    FreqIncr=np.fromfile(file, count=1,dtype= np.dtype('<d'), offset = offset)[0]   # [GHz] Incremento de frecuencia  OJO!!! Para las medidas en frecuencias
    offset += np.dtype('<d').itemsize

    FI=FreqIncr

    StartTime=np.fromfile(file, count=1,dtype= np.dtype('<d'),offset = offset)[0]   # [ns] Tiempo de inicio de medida. Representa cuando empieza la medida de la fibra exterior al OBR
    offset += np.dtype('<d').itemsize

    TimeIncr=np.fromfile(file, count=1,dtype= np.dtype('<d'),offset = offset)[0]    # [ns] Incremento de tiempo.
    offset += np.dtype('<d').itemsize

    dt=TimeIncr                                                                     # [ns] Incremento de tiempo=delta de tiempo.

    #MeasurementType=np.fromfile(file, count=1,dtype= 'uint16',offset = offset)[0]   # Cero si es reflexi�n
    offset += np.dtype('uint16').itemsize

    GroupIndex=np.fromfile(file, count=1,dtype= np.dtype('<d'),offset = offset)[0]   # n de la fibra. Alrededor de 1.5
    offset += np.dtype('<d').itemsize

    #GainValue=np.fromfile(file, count=1,dtype= 'int32',offset = offset)[0]           # Ganancia (dB)
    offset += np.dtype('int32').itemsize

    #ZeroLengthIndex=np.fromfile(file, count=1,dtype= 'int32',offset = offset)[0]     #(StartTime)/(Time increment) Columna que le corresponde al inicio de medida de la fibra exterior al OBR
    offset += np.dtype('int32').itemsize

    #DataTypeSize=np.fromfile(file, count=1,dtype= 'uint32',offset = offset)[0]       #Numero de bytes de cada punto (8)
    offset += np.dtype('uint32').itemsize

    nPoints=np.fromfile(file, count=1,dtype= 'uint32',offset = offset)[0]            #Numero de puntos del Array (Real + Imaginario)
    offset += np.dtype('uint32').itemsize

    n=int(nPoints/2)                                                                 #n es el numero de puntos correspondientes a cada medida

    DateTime=np.fromfile(file, count=8,dtype= 'uint16',offset = offset)              #Fecha de la medida
    offset += np.dtype('uint16').itemsize * 8

    CalibrationDate=np.fromfile(file, count=8,dtype= 'uint16',offset = offset)       #Fecha de la calibraci�n
    offset += np.dtype('uint16').itemsize * 8

    #TempCoeffs=np.fromfile(file, count=5,dtype= np.dtype('<d'),offset = offset)      #Coeficientes de la conversi�n en temperatura
    offset += np.dtype('<d').itemsize * 5

    #StrainCoeffs=np.fromfile(file, count=5,dtype= np.dtype('<d'),offset = offset)    #Coeficientes de la conversi�n en deformaciones
    offset += np.dtype('<d').itemsize * 5

    #FreqWinFlg=np.fromfile(file, count=1,dtype= 'uint8',offset = offset)[0]          #1 Si se le ha aplicado el filtro en frecuencias. Cero si no se ha hecho
    offset += np.dtype('uint8').itemsize

    #Unused=np.fromfile(file, count=1865,dtype= 'uint8',offset = offset)              #Sin uso
    offset += np.dtype('uint8').itemsize * 1865

    Preal=np.fromfile(file, count=n,dtype= np.dtype('<d'),offset = offset)     #Medida de la polarización p Real (en función del tiempo)
    offset += np.dtype('<d').itemsize * n
    Pimag=np.fromfile(file, count=n,dtype= np.dtype('<d'),offset = offset)     #Medida de la polarización p Imaginaria
    offset += np.dtype('<d').itemsize * n
    Sreal=np.fromfile(file, count=n,dtype= np.dtype('<d'),offset = offset)     #Medida de la polarización s Real
    offset += np.dtype('<d').itemsize * n
    Simag=np.fromfile(file, count=n,dtype= np.dtype('<d'),offset = offset)     #Medida de la polarización s Imaginaria
    offset += np.dtype('<d').itemsize * n

    # Device=np.fromfile(file, count=1, dtype= '|S1', offset = offset)[0].astype('|U8')  #Depende del nombre del archivo

    #fclose(file)

    """ Posterior calculations """

    GroupIndex = 1.4682

    tf=(n-1)*dt                            #f es el valor final de tiempo hasta el que se mide, calculado a traves del numero de puntos del array (n)
    t=np.arange(0,tf+dt,dt)                #t es un array que tiene un incremento lineal delta de tiempo hasta el tiempo final de medida
    t=t+StartTime
    z=t*((299792458e-9)/(GroupIndex*2))    #[m]     -> tiempo en nanosegundos, z en metros 299792458 es la velocidad de la luz en el vacio y en indice de grupo es el indice de refracci�n medio de la fibra
    f=np.arange(SF-FI*2*n,SF+FI,FI)        #[GHz]   -> f es un array que tiene un incremento lineal (incremento de frecuencia) desde la frecuencia de inicio hasta la última

    ### Calculations

    Pc = Preal+Pimag*1j     #Pc es la medida de la polarización p en complejos
    Sc = Sreal+Simag*1j     #Sc es la medida de la polarización s en complejos
    St = stokes_vector(P,S)

    return f,z,Pc,Sc,St

def multi_read_obr(files,path_to_data='.',limit1 = 'none',limit2 = 'none',display=False):
    """ Function to read multiple obr files and crop it
    after displaying all signals in the same plot (if display == True)

    :param files        (list)      : List of files to be read (without .orb extension)
    :param path_to_data (string)    : Path to directory which contains all obr files
    :param limit1       (float)     : First length ofto conserve in data arrays
    :param limit2       (float)     : Last length to conserve in data arrays
    :param display      (boolean)   : Boolean to display read data (True) or not (False)

    :return f,z,Data          : Read "read_obr" above
                                        Data = [Pc,Sc,Hc]
                                        f and z are from the last lecture (asuming they are the same for all lectures)
    """

    Data = dict()
    pending_files = list()
    print('*Start reading')


    if display == True:
        max_plots = min(3,len(files)-1)
        plt.figure()
        for idx,file in enumerate(files):
            print('Reading',file)
            if idx < max_plots:
                f,z,Pc,Sc,Hc =read_obr(f'{path_to_data}/{file}.obr')
                plt.plot(z,np.log10(np.absolute(Hc)))
                Data[file] = [Pc,Sc,Hc]
                pending_files.append(file)
            elif idx == max_plots:
                f,z,Pc,Sc,Hc =read_obr(f'{path_to_data}/{file}.obr')
                plt.plot(z,np.log10(np.absolute(Hc)))
                Data[file] = [Pc,Sc,Hc]
                pending_files.append(file)
                plt.xlabel('z [m]')
                plt.ylabel(r'$Log_{10}(H(t))$')
                plt.grid()
                plt.show()
                # Ask for limits
                if limit1 == 'none':
                    try:
                        limit1 = float(input('First limit:'))
                    except:
                        limit1 = 0
                if limit2 == 'none':
                    try:
                        limit2 = float(input('Second limit:'))
                    except:
                        limit2 = -1
                limit1 = find_index(z,limit1)
                limit2 = find_index(z,limit2)
                # Crop previous data
                for file in pending_files:
                    for measure in range(len(Data[file])):
                        Data[file][measure] = Data[file][measure][int(limit1):int(limit2)]
                f = f[int(limit1):int(limit2)]
                z = z[int(limit1):int(limit2)]

            elif idx > max_plots:
                f,z,Pc,Sc,Hc =read_obr(f'{path_to_data}/{file}.obr')
                Data[file] = [Pc,Sc,Hc]
                for measure in range(len(Data[file])):
                    Data[file][measure] = Data[file][measure][int(limit1):int(limit2)]
                f = f[int(limit1):int(limit2)]
                z = z[int(limit1):int(limit2)]
            else:
                print('Error')

        print('*End reading')
    else:
        if limit1 == 'none':
            try:
                limit1 = float(input('First limit:'))
            except:
                limit1 = 0
        if limit2 == 'none':
            try:
                limit2 = float(input('Second limit:'))
            except:
                limit2 = -1


        for idx,file in enumerate(files):
            print('Reading',file)
            f,z,Pc,Sc,Hc =read_obr(f'{path_to_data}/{file}.obr')
            Data[file] = [Pc,Sc,Hc]
            if idx == 0:
                limit1 = find_index(z,limit1)
                limit2 = find_index(z,limit2)

            for measure in range(len(Data[file])):
                Data[file][measure] = Data[file][measure][int(limit1):int(limit2)]

        f = f[int(limit1):int(limit2)]
        z = z[int(limit1):int(limit2)]


        print('*End reading\n')


    return f,z,Data

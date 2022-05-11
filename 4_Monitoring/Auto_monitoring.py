import time
import os
import pandas as pd
import numpy as np


import pyautogui
import serial


""" SETTINGS """

USER = 'LUNA' #'temis'
path_to_obr = r'C:\Users\Luna\Desktop\Andres\Data\Acero_simetrico\0_OBR'

total_time = 240*60 # seconds
period     = 1*60   # seconds
measures_time = np.arange(0, total_time+period, period)

global exception_number
global timeout_number

exception_number = 0
timeout_number   = 0

# Profiles
if USER == 'temis':
    """
    Please do the following steps before running this code
        1. Upload thermocouple.ino to the Arduino board which is connected in port "arduino_port", identify it
        2. Open LUNA OBR v3.13.0 software, configure it to get the proper measurements (sensing enable)
        3. Position "Fast scan" so it is clickable on X=286, Y=483
    """
    ser = serial.Serial('/dev/ttyUSB0', 9600)
    LUNAobr_WindowName = 'Desktop1'
    X = 289
    Y = 483

elif USER == 'LUNA':
    """
    Please do the following steps before running this code
        1. Upload thermocouple.ino to the Arduino board which is connected in port "arduino_port", identify it
        2. Open LUNA OBR v3.13.0 software, configure it to get the proper measurements (sensing enable)
        3. Position "Fast scan" so it is clickable on X=200, Y=385 and in fisrt plane
    """
    #ser = serial.Serial('COM4', 9600)
    LUNAobr_WindowName = 'OBR'
    X = 200
    Y = 385


""" FUNCTION DEFINITIONS """

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
    # Print New Line on Complete
    if iteration == total:
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\n')


def save_data(filename):
    """ With Luna's OBR software opened clicks to save data with filename specified"""
    
    global timeout_number
    global exception_number

    # Go to Luna's OBR window
    if USER == 'temis':
        os.system(f'wmctrl -a {LUNAobr_WindowName}')  # in my case I'm using a remmina remote desktop from linux
    elif USER == 'LUNA':
        """
        from pywinauto import application
        app = application.Application()
        app.start(LUNAobr_WindowName)
        app.UntitledNotepad.SetFocus()
        """
        """
        something went wrong and I weren't able to switch between windows so OBR program must be oppened at side
        """
        pass

    # Click on fast scan (X=289,Y=483)
    pyautogui.moveTo(X,Y, duration=0)
    time.sleep(1)
    pyautogui.click()
    pyautogui.click()
    pyautogui.click()
    print('Performing measurement...')
    time.sleep(10) #time.sleep(4.2) # 4 seconds for fast scan, about 18 seconds for normal scan
    print('Measurement done')
    pyautogui.moveTo(X+50,Y, duration=0)
    pyautogui.click()
    # Stroke CTRL+S to save data
    time.sleep(1)
    pyautogui.hotkey('ctrl', 's')
    # Write filename
    time.sleep(2)
    pyautogui.write(filename,interval=0.0)
    # Press enter
    pyautogui.press('enter')
    time.sleep(2)
    # Overwrite if needed
    try:
        pyautogui.press('enter') # Overwritte
        time.sleep(2)
        pyautogui.press('enter') # Binary data incomplete
        time.sleep(2)
        pyautogui.press('enter') # Warning
    except Exception as e:
        pyautogui.screenshot(f'exception_{exception_number}.png'); exception_number = exception_number + 1
        pass


def take_measure(idx,label):
    
    global timeout_number
    global exception_number
    
    print('')
    print(f' Taking measure {label}')
    measure_taken = False
    timeout = 20 # s
    zero_time = float(time.time())

    try:
        while measure_taken == False:

            # Save OBR data
            filename = f'{int(idx)}_s_{label}' if isinstance(idx, (float,int)) else  f'{idx}_{label}'
            save_data(filename)
            measure_taken = check_measure(f'{path_to_obr}\{filename}.obr')
            print('     OBR saved!')

            # Save data to csv
            df = pd.DataFrame({'idx': [idx], 'filename': filename ,'global time': [ time.strftime("%Y,%m,%d,%H:%M:%S")]})
            with open(path, 'a') as f:
                df.to_csv(f, header=False, index=False)
            print('     CSV saved!')

            # Check timeout
            if float(time.time())-zero_time >= timeout:
                print('timeout')
                pyautogui.screenshot(f'timeout_{timeout_number}.png'); timeout_number = timeout_number + 1
                return False
                break

        return True

    except Exception as e:
        print(e)
        pyautogui.screenshot(f'exception_{exception_number}.png'); exception_number = exception_number + 1
        return False

def check_measure(file):
        """
            Function to read binaries (OBR)

            param
                file    (string): path to file to check

            returns
                True if the  current datetime of measure is the same as specified in .obr file saved just before
                False if the current datetime is different
        """
        
        global timeout_number
        global exception_number

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

        try:
            DateTime=np.fromfile(file, count=8,dtype= 'uint16',offset = offset)              #Fecha de la medida
            current_time = np.array(time.localtime()[0:-1])

            if DateTime[0] == current_time[0] and DateTime[1] == current_time[1] and  DateTime[3] == current_time[2] and  DateTime[4] == current_time[3] and abs(DateTime[5]-current_time[4]) <= 1:
                return True
            else:
                print(DateTime)
                print(current_time)
                return False

        except Exception as e:
            pyautogui.screenshot(f'exception_{exception_number}.png'); exception_number = exception_number + 1
            print(e)
            return False

# Create dataframe with times and temperatures

path = './recording.csv'
if not os.path.exists(path):
    df = pd.DataFrame(columns=['idx', 'filename','global time'])
    df.to_csv(path, index=False)

""" PROCESS """

# Check out
print('')
print('*** Summary ***')
print('')
print(f'    Total time:       {total_time/60} min or {total_time} s ')
print(f'    Period:           {period/60} min or {period} s ')
print('')

# Start recording
print('*** Start recording ***')
input('Press enter to start')
print('')

take_measure('prueba', 'prueba')
zero_time    = float(time.time())
elapsed_time = float(time.time()) - zero_time
timeout      = period-10 # seconds

for measure_time in measures_time:
    

    print(f'*Current measure time {measure_time/60} min or {measure_time} s')
    print('')

    # zero_times
    measure_zero_time = timeout_zero_time = float(time.time())

    try:

        performed1   = False
        performed2   = False

        while performed1==False and performed2==False:
            

            # Get total elapsed time
            total_elapsed_time = float(time.time()) - zero_time

            # Progress bar on waiting
            try:
                PB_current = measure_elapsed_time = float(time.time()) - measure_zero_time
                PB_total   = period
                printProgressBar(PB_current , PB_total, prefix = ' Progress:', suffix = 'Complete', length = 50)
            except Exception as e:
                pyautogui.screenshot(f'exception_{exception_number}.png'); exception_number = exception_number + 1
                print(e)

            # Take first measure at time
            if total_elapsed_time>=measure_time and performed1 == False:

                performed1 = take_measure(total_elapsed_time,'measure_1')

            # Take second measure more than 10 seconds later
            if total_elapsed_time>=measure_time+10 and performed2 == False:

                performed2 = take_measure(total_elapsed_time,'measure_2')

            # Break while if timeout
            if (float(time.time())-timeout_zero_time) >= timeout:
                pyautogui.screenshot(f'timeout_{timeout_number}.png'); timeout_number = timeout_number + 1
                break

        print('')

        # If no measure was performed take one
        if performed1==False and performed2==False:
            print(f'*** No measure taken for meausure time: {measure_time} ***')
            total_elapsed_time = float(time.time()) - zero_time
            performed1 = performed2 = take_measure(total_elapsed_time,'no_measure')
        else:
            print(f'Measures for measure time = {measure_time} s done!')

    except Exception as e:
        pyautogui.screenshot(f'exception_{exception_number}.png'); exception_number = exception_number + 1
        print(e)
        
        try:
            print(f'*** Error getting measure for measure time: {measure_time} s ***')
            total_elapsed_time = float(time.time()) - zero_time
            take_measure(total_elapsed_time,'except')

        except Exception as e:
            pyautogui.screenshot(f'exception_{exception_number}.png'); exception_number = exception_number + 1
            print(e)
            print('*** Something went really wrong :( ***')


    print('')

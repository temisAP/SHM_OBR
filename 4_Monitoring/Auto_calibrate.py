import time
import os
import re
import pandas as pd
import numpy as np
import subprocess


import pyautogui
import serial


""" SETTINGS """

USER = 'LUNA' #'temis'
path_to_obr = r'C:\Users\Luna\Desktop\Andres\Data\Calibration'

segments_temp = [30,180,185,190,195,200] #180 + np.array([0,5,10,15,20,25]) # °C
segments_time = [ 2,  2, 2,  2, 2,  2]      # min
measure1_time = 1                           # min
measure2_time = 1.2                         # min

segments_time = 60 * np.array(segments_time) # min to s
measure1_time = 60 * measure1_time           # min to s
measure2_time = 60 * measure2_time           # min to s

weight = 0e-3 * 9.81 # N
b = 30.4e-3           # m
t = 2e-3              # m
L = 300e-3            # m
I = 1/12 * b * t**3   # m⁴
E = 70.3e9              # Pa
flecha = weight * L**3 / (3*E*I)
flecha = flecha * 1e3

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
    ser = serial.Serial('COM4', 9600)
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
    pyautogui.moveTo(X,Y, duration=0)
    # Stroke CTRL+S to save data
    pyautogui.hotkey('ctrl', 's')
    # Write filename
    time.sleep(1)
    pyautogui.write(filename,interval=0.0)
    # Press enter
    pyautogui.press('enter')
    time.sleep(1)
    # Overwrite if needed
    try:
        pyautogui.press('enter')
    except Exception as e:
        pass

def get_current_temperature(verbosity=False):

    zero_time = float(time.time())
    timeout = 20 # s

    read = False

    while read == False:
        try:
            # read line
            line = ser.readline()
            # decode bits to string
            line = line.decode('unicode_escape')
            # remove newline
            line = line.rstrip()
            args = line.split(' ')
            for arg in args:
                if '°' in arg:
                    temperature = float(arg.split('°')[0])
            if verbosity == True:
                print(f'{line} ---> Temperature: {temperature} °C')
        except Exception as e:
            print(e)
            temperature = 'error'
            line = 'reading error'

        # Check if temperature has been taken
        if isinstance(temperature, float):
            read = True
        else:
            print(f'{line} ---> Temperature: {temperature} °C')
            time.sleep(1)

        # Check timeout
        try:
            if (float(time.time())-zero_time) >= timeout:
                time_out = True
                temperature = 0
                print('Timeout')
                break
        except Exception as e:
            print(e)
            print('Error computing timeout')

    return float(temperature)

def take_measure(idx,label):
    print('')
    print(f' Taking measure {label}')
    measure_taken = False
    timeout = 30
    zero_time = float(time.time())

    try:
        while measure_taken == False:
            # Get temperature from segment array
            try:
                temperature = get_current_temperature()
                print('     Current temperature =',temperature,'C')
            except Exception as e:
                print(e)
                return False

            # Save OBR data
            filename = f'{flecha}_mm_{temperature:.2f}_grados_{label}_{idx}'
            save_data(filename)
            measure_taken = check_measure(f'{path_to_obr}\{filename}.obr')
            print('     OBR saved!')

            # Save data to csv
            df = pd.DataFrame({'idx': [idx], 'filename': filename, 'temperature [C]': [temperature], 'flecha [mm]':[flecha] ,'global time': [ time.strftime("%Y,%m,%d,%H:%M:%S")]})
            with open(path, 'a') as f:
                df.to_csv(f, header=False, index=False)
            print('     CSV saved!')

            # Check timeout
            if float(time.time())-zero_time >= timeout:
                print('timeout')
                return False
                break

        return True

    except Exception as e:
        print(e)
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
            print(e)
            return False

# Create dataframe with times and temperatures

path = './recording.csv'
if not os.path.exists(path):
    df = pd.DataFrame(columns=['idx', 'filename', 'temperature [C]', 'flecha [mm]','global time'])
    df.to_csv(path, index=False)

""" PROCESS """

# Check out
print('')
print('*** Summary ***')
print(f'The calibration will consist on {len(segments_temp)} segments')

idx =0
for segment_temp,segment_time in zip(segments_temp,segments_time):
    print(f' Segment no{idx}:')
    print(f'    segment temperaure: {segment_temp} ºC')
    print(f'    segment time:       {segment_time/60} min or {segment_time} s ')
    idx += 1
print('')
print(f'Performing calibration with a {weight/(1e-3 * 9.81)} g weight -> {flecha} mm')
print('')

# Start recording
print('*** Start recording ***')
input('Press enter to start')
print('')

take_measure(99, 'prueba')


idx=0
last_segment_temp = get_current_temperature()

for segment_temp,segment_time in zip(segments_temp,segments_time):

    print(f'* Segment no{idx}, at {segment_temp} ºC')
    print('')

    try:

        # Ramp
        print(' Waiting on ramp')
        last_time = float(time.time())
        current_temp = last_temp = get_current_temperature()

        while current_temp < segment_temp-0.5 or current_temp > segment_temp+0.5:

            try:
                # Get temperature evolution
                current_temp = get_current_temperature()
                current_time = float(time.time())
                dT = (current_temp - last_temp)/(current_time - last_time)
                last_temp = current_temp
                last_time = current_time
                if abs(current_temp-last_temp) > 30:
                    print('Difference between current and last temperature bigger than 30')
                    print('The measuere will be discarded')
                    current_temp = last_temp
                # Update progressbar
                PB_total =   abs(segment_temp-last_segment_temp)
                PB_current = abs(current_temp-last_segment_temp)
                printProgressBar(PB_current , PB_total, prefix = ' Progress:', suffix = f'{current_temp:.2f}/{segment_temp:.2f}°C   Estimated remaining time {((segment_temp-current_temp)/dT/60):.2f} min', length = 50)

            except Exception as e:
                print(e)
                try:
                    current_temp = get_current_temperature(verbosity=True)
                    if abs(current_temp-last_temp) > 30:
                        print('Difference between current and last temperature bigger than 30 K')
                        print('The measuere will be discarded')
                        current_temp = last_temp
                except Exception as e:
                    print(e)
                    print('Error computing ramp and getting temperature')


        print('')
        print('')

        # Segment
        print(' Waiting to stabilization')
        segment_zero_time = float(time.time())
        segment_elapsed_time = segment_zero_time
        performed1 = False
        performed2 = False
        segment_warning = False

        while (segment_elapsed_time-segment_zero_time <= segment_time) or (current_temp >= segment_temp-0.5 and current_temp <= segment_temp+0.5):


            try:
                # Update progress bar
                current_temp = get_current_temperature()
                printProgressBar(segment_elapsed_time-segment_zero_time, segment_time, prefix = ' Progress:', suffix = f'{(segment_elapsed_time-segment_zero_time)/60:.2f}/{segment_time/60} min  Current temp: {current_temp} ºC', length = 50)
            except Exception as e:
                print(e)
                try:
                    current_temp = get_current_temperature(verbosity=True)
                except Exception as e:
                    print(e)
                    print('Error computing segment and getting temperature')


            # Take first measure when 60 seconds remain on segment
            if (segment_elapsed_time-segment_zero_time >= measure1_time) and performed1 == False:

                performed1 = take_measure(idx,'1')

            # Take first measure when 30 seconds remain on segment
            if (segment_elapsed_time-segment_zero_time >= measure2_time) and performed2 == False:

                performed2 = take_measure(idx,'2')

            # Warning
            if (segment_elapsed_time-segment_zero_time > segment_time) and (current_temp >= segment_temp-0.5 and current_temp <= segment_temp+0.5) and segment_warning == False:
                print('Segment timeout but temperature still stable: waiting on segment')
                segment_warning = True

            segment_elapsed_time=float(time.time())

        print('')

        # If no measure was performed take one
        if performed1==False and performed2==False:
            print(f'*** No measure taken in segment: {idx} ***')
            take_measure(idx,'no_measure')
            performed1 = True
        else:
            print(f'Segment no{idx} done!')

        idx=idx+1

    except Exception as e:
        print(e)
        try:
            print(f'*** Error getting measure in segment: {idx} ***')
            take_measure(idx,'except')

            # Update cycle
            last_segment_temp = segment_temp
            idx=idx+1

        except Exception as e:
            print(e)
            print('*** Something went really wrong :( ***')
            if performed1 == True or performed2 == True:
                # Update cycle
                last_segment_temp = segment_temp
                idx=idx+1


    print('')

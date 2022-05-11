import os

folder = './data'

files = os.listdir(folder)

offset = 2000

for file in files:
    if file.endswith('.obr'):
        time  = float(file.split('_')[0])
        new_name = file.replace(f'{int(time)}',f'{int(time+offset)}')
        print(file,'\t -> \t',new_name)
        os.rename(f'{folder}/{file}', f'{folder}/{new_name}')

import os
import random
import time
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import psutil


class the_dataset(Dataset):
    """ Signal dataset """
    def __init__(self, X, Y, transform=None):
        self.x = torch.from_numpy( np.array(X) ).float()
        self.y = torch.from_numpy( np.array(Y) ).float()
        self.N = Y.shape[0] # len(Y[:][0]) = 2 ; len(Y) = n_samples

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.x[idx,:], self.y[idx,:]


def load_datasets(self,
                    datasets = None, ds_percentages = 100,
                    split = True, test_percentage = 20, val_percentage = 20,
                    preprocessing = True, plot_preprocessing = False, plot_histogram = False,
                    data_loaders = True):

    """ Function to load data from datasets

        :param: datasets(list of strings): path of datasets to load -> If None a GUI to select will be launched


        :optional: ds_percentages (list of floats)  : percentage of a dataset to append to full dataset
        :optional: train_percentage (float)         : percentage of full dataset to consider as training data
        :optional: test_percentage (float)          : percentage of full dataset to consider as test data
        :optional: validation_percentage (float)    : percentage of full dataset to consider as validation data

    """

    """ Check out """

    # GUI option
    if datasets is None:
        print('Please specify a dataset')
        return # Y aquí una función con GUI para identificar dónde está

    # If dataset is an string makes them a list with one element
    if not isinstance(datasets, list):
        datasets = [datasets]
    if not isinstance(ds_percentages, list):
        ds_percentages = np.ones(len(datasets))  * 100.0

    if len(self.X) != 0:
        ans = input('There are currently data storaged (append/overwrite/quit): ')
        if 'a' in ans:
            pass
        if 'o' in ans:
            self.clear_dataset(auto=True) # To free RAM space self.X and self.Y now are empty
            time.sleep(10)
        if 'q' in ans:
            return
    elif len(self.X) == 0:
        print('\nIA dataset is empty')
        pass

    """ Load datasets """

    self.get_datasets(datasets,ds_percentages)

    """ Train test split """

    self.split_datasets(test_percentage, val_percentage) if split else False

    """ Normalization """

    self.pre_processing(plot_preprocessing = plot_preprocessing, plot_histogram = plot_histogram) if preprocessing else False

    """ Data loaders """

    if data_loaders and split:
        self.get_dataloaders()
    else:
         self.dl = None

    print('')

def get_datasets(self,datasets,ds_percentages=[100]):
    # Change dataset dict to list
    self.X = [item for key,sublist in self.X.items() for item in sublist]
    self.Y = [item for key,sublist in self.Y.items() for item in sublist]

    # Join all datasets in one
    for dataset,percentage in zip(datasets,ds_percentages):
        print('\nLoading',dataset)

        # Load new data
        newX, newY = load_one_dataset(dataset)

        # Extend dataset with new value
        self.X.extend(newX)
        self.Y.extend(newY)

def load_one_dataset(dataset):
    """ Load a dataset storaged with pickle """

    # Check if path is right
    if not os.path.exists(dataset):
        print('\n','Dataset not found in:')
        print(dataset)
        return

    # Import dataset
    class ds(object):
        def __init__(self,path,name):
            self.path   = path
            self.name   = name
            self.load()
            print(f'\nDATASET with {len(self.X)} elements loaded!')

        from .load import load

    ds_obj = ds(os.path.dirname(dataset),os.path.basename(dataset))

    return ds_obj.X, ds_obj.Y

def split_datasets(self,test_percentage = 20, val_percentage = 20):

    # Absolute values
    test_percentage  = test_percentage/100
    val_percentage   = val_percentage /100
    train_percentage = 1 - test_percentage - val_percentage
    train_val_percentage = 1 - test_percentage

    # Validation percentage after splitting train and test
    val_percentage          = val_percentage/train_val_percentage
    train_percentage        = train_percentage/train_val_percentage

    # Split datasets in train test and validation
    if val_percentage != 0:
        percentages = {
                    'train' :train_percentage,
                    'test'  :test_percentage,
                    'val'   :val_percentage}
    elif val_percentage == 0:
        percentages = {
                    'train' :train_percentage,
                    'test'  :test_percentage}

    X = dict.fromkeys(percentages.keys())
    y = dict.fromkeys(percentages.keys())

    # Train test dataset split
    X['train'], X['test'], y['train'], y['test']  = train_test_split(self.X, self.Y, test_size = percentages['test'], random_state=1)

    # To free RAM space self.X and self.Y now are empty
    self.clear_dataset(auto=True)
    zero_time = float(time.time())
    while psutil.virtual_memory()[2] > 90:
        print('Waiting for memory')
        time.sleep(1)
        if float(time.time())-zero_time > 60:
            exit()

    # Create a validation set
    if val_percentage != 0:
        X['train'], X['val'], y['train'], y['val']  =  train_test_split(X['train'], y['train'], test_size = percentages['val'], random_state=1)

    self.X = X
    self.Y = y

def get_dataloaders(self):
    # Dict of dataloaders
    dl = dict.fromkeys(self.X.keys())
    for key, val in dl.items():
        dl[key] = DataLoader(
                    dataset=the_dataset(self.X[key],self.Y[key]),
                    batch_size=32,
                    shuffle=True)
        self.X[key] = 'In data loader'
        self.Y[key] = 'In data loader'

    # Re asign to object atribute
    self.dl = dl

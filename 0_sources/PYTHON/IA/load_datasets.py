import os
import random
import time
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class the_dataset(Dataset):
    """ Signal dataset """
    def __init__(self, X, Y, transform=None):
        self.x = torch.from_numpy( np.array(X) ).float()
        self.y = torch.from_numpy( np.array(Y) ).float()
        self.N = len(Y) # len(Y[:][0]) = 2 ; len(Y) = n_samples

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.x[idx,:], self.y[idx,:]


def load_datasets(self,datasets=None,ds_percentages=100,split = True, preprocessing=True, data_loaders = True, train_percentage=60,test_percentage=40,val_percentage=40,plot_preprocessing=False,plot_histogram=False):
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

    # Change dataset dict to list
    self.X = [item for key,sublist in self.X.items() for item in sublist]
    self.Y = [item for key,sublist in self.Y.items() for item in sublist]

    # Join all datasets in one
    for dataset,percentage in zip(datasets,ds_percentages):
        print('\nLoading',dataset)

        # Load new data
        newX, newY = load_one_dataset(dataset)

        # Reduce data
        newX = random.sample(newX,int(len(newX)*percentage/100))
        newY = random.sample(newY,int(len(newY)*percentage/100))

        # Extend dataset with new value
        self.X.extend(newX)
        self.Y.extend(newY)

    """ Train test split """

    if split:

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

        X['train'], X['test'], y['train'], y['test']  = train_test_split(self.X, self.Y, test_size = percentages['test']/100, random_state=1)

        self.clear_dataset(auto=True) # To free RAM space self.X and self.Y now are empty
        time.sleep(10)

        if val_percentage != 0:
            X['train'], X['val'], y['train'], y['val']  =  train_test_split(X['train'], y['train'], test_size = percentages['val']/100, random_state=1) # 0.25 x 0.8 = 0.][
    else:
        X = self.X
        y = self.Y

    """ Normalization """

    # Preprocessing
    if preprocessing:
        X, y = self.pre_processing(X,y,plot_preprocessing = plot_preprocessing, plot_histogram = plot_histogram)

    """ Data loaders """


    # Dict of dataloaders
    if data_loaders and split:
        dl = dict.fromkeys(X.keys())
        for key, val in dl.items():
            dl[key] = DataLoader(
                        dataset=the_dataset(X[key],y[key]),
                        batch_size=32,
                        shuffle=True)
    else:
        dl = None

    # Re asign to object atribute
    self.X = X
    self.Y = y
    self.dl = dl

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
            print('\nDATASET loaded!')

        from .load import load

    ds_obj = ds(os.path.dirname(dataset),os.path.basename(dataset))

    return ds_obj.X, ds_obj.Y

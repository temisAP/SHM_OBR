import time
from statistics import mean
import numpy as np

import torch
import torch.nn as nn
from .model import TE

from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve



def fit_data(self,num_epochs = 25, lr=1e-7, representation = True, criterion  = nn.MSELoss(),save=False,cv=2):

    print("\nFitting data\n")

    if not hasattr(self, "model"):
        self.model = TE()

    if isinstance(self.model, torch.nn.Module):
        self.model = fit_torch_model(self, num_epochs = num_epochs,lr=lr, representation=representation, criterion=criterion)
    else:
        self.model = fit_sklearn_model(self, representation=representation, save=save, cv=cv)


def fit_torch_model(obj,num_epochs=25,lr=1e-7, representation= True, criterion=nn.MSELoss()):

    # Load models
    device = obj.device
    model  = obj.model.to(device)

    # Optimizers
    optimizer  = torch.optim.Adam(model.parameters(),lr=lr)

    # Train model

    train_loss_T = list()
    train_loss_E = list()
    valid_loss_T = list()
    valid_loss_E = list()

    train_loss = []
    val_loss = []
    zero_time = float(time.time())

    for epoch in range(num_epochs):
        loss_sum = 0
        train_loss_T_sum = 0
        train_loss_E_sum = 0
        model.train()
        for (x, y) in obj.dl['train']:
            x = x.to(device)
            y = y.to(device)
            # Forward pass
            T, E = model(x)
            # Losses
            loss_T = criterion(T, y[:,0])
            loss_E = criterion(E, y[:,1])
            loss = loss_T + loss_E
            loss_sum += loss
            train_loss_T_sum += loss_T/len(y[:,0])
            train_loss_E_sum += loss_E/len(y[:,1])
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        else:

            with torch.no_grad():
                model.eval()
                loss_sum = 0
                valid_loss_T_sum = 0
                valid_loss_E_sum = 0
                for x, y in obj.dl['val']:
                    x = x.to(device)
                    #forward pass
                    T, E = model(x)
                    # Mean Square Error
                    valid_loss_T_sum += torch.sum( torch.abs( T.to('cpu') - y[:,0].to('cpu') ) )/len(y[:,0])
                    valid_loss_E_sum += torch.sum( torch.abs( E.to('cpu') - y[:,1].to('cpu') ) )/len(y[:,1])

                # Times
                elapsed_time        = (float(time.time())-zero_time)
                time_per_epoch      = elapsed_time/(epoch+1)
                remaining_epochs    = num_epochs - epoch
                remaining_time      = time_per_epoch * remaining_epochs

                # Information
                train_loss_T.append(float(train_loss_T_sum.cpu().detach().numpy()))
                valid_loss_T.append(float(valid_loss_T_sum.cpu().detach().numpy()))
                train_loss_E.append(float(train_loss_E_sum.cpu().detach().numpy()))
                valid_loss_E.append(float(valid_loss_E_sum.cpu().detach().numpy()))
                                                                                                     

                print(f'Epoch: {epoch+1} of {num_epochs} || Remaining time: {time.strftime("%H:%M:%S",  time.gmtime(remaining_time))}')
                print(f' Training T loss: {train_loss_T_sum:.4f} || Validation T loss: {valid_loss_T_sum:.4f} ')
                print(f' Training E loss: {train_loss_E_sum:.4f} || Validation E loss: {valid_loss_E_sum:.4f} ')

    if representation:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(train_loss_T,label='train loss (T)')
        plt.plot(train_loss_E,label='train loss (E)')
        plt.plot(valid_loss_T,label='validation loss (T)')
        plt.plot(valid_loss_E,label='validation loss (E)')
        plt.legend()
        plt.grid()
        plt.show()

    return model


def fit_sklearn_model(IA_obj,representation=True,save=False,cv=3):

    IA_obj.model_T.fit(IA_obj.T['train'], IA_obj.E['train'][:,0])
    IA_obj.model_E.fit(IA_obj.T['train'], IA_obj.E['train'][:,1])

    IA_obj.save_model()

    train_sizes_T, train_scores_T, valid_scores_T = learning_curve(
            IA_obj.model_T, IA_obj.T['train'], IA_obj.E['train'][:,0],cv=cv)

    train_sizes_E, train_scores_E, valid_scores_E = learning_curve(
            IA_obj.model_E, IA_obj.T['train'], IA_obj.E['train'][:,1],cv=cv)

    if representation or (not save == False):
        import matplotlib.pyplot as plt
        import matplotlib
        cmap = matplotlib.cm.get_cmap('tab10')

        plt.figure()
        for i in range(int(cv*2)):
            if i%2 == 0:
                plt.plot(train_scores_T[i],'-' ,label=f'train scores (T) {i} ({train_sizes_T[i]} elements)',color=cmap(i))
                plt.plot(valid_scores_T[i],'--',label=f'valid scores (T) {i}',color=cmap(i))
            else:
                plt.plot(train_scores_E[i],'-' ,label=f'train scores (E) {i} ({train_sizes_E[i]} elements)',color=cmap(i))
                plt.plot(valid_scores_E[i],'--',label=f'valid scores (E) {i}',color=cmap(i))
        plt.legend()
        plt.grid()
        plt.show() if representation else False
        plt.savefig(f'{save}_training.png') if not save == False else False

    return IA_obj.model_E, IA_obj.model_T

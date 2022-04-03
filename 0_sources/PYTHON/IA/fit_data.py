from statistics import mean
import torch
import torch.nn as nn
import time
from .model import splitter
import numpy as np

def fit_data(self,num_epochs = 25, lr=1e-7, representation = True, criterion  = nn.MSELoss()):

    print("\nFitting data\n")

    # Load models
    device = self.device
    self.model_T = splitter() # Model for temperature extraction
    self.model_E = splitter() # Model for deformation extraction
    model_T  = self.model_T.to(device)
    model_E  = self.model_E.to(device)

    # Optimizers
    optimizer_T  = torch.optim.Adam(model_T.parameters(),lr=lr)
    optimizer_E  = torch.optim.Adam(model_E.parameters(),lr=lr)

    # Train model

    train_loss_T = list()
    train_loss_E = list()
    validation_loss_T = list()
    validation_loss_E = list()

    zero_time = float(time.time())

    for epoch in range(num_epochs):

        loss_T_sum = 0
        val_loss_T_sum = 0
        loss_E_sum = 0
        val_loss_E_sum = 0

        # Training
        model_T.train()
        model_E.train()

        # Evaluation
        for (x, y) in self.dl['train']:

            # To device
            x = x.to(device)
            y = y.to(device)
            # Clear the gradients
            optimizer_T.zero_grad()
            optimizer_E.zero_grad()
            # Forward pass
            T = model_T(x)
            E = model_E(x)
            # Losses
            loss_T = criterion(T, y[:,0].view(-1,1))
            loss_E = criterion(E, y[:,1].view(-1,1))
            loss_T_sum += loss_T/len(y[:,0])
            loss_E_sum += loss_E/len(y[:,1])
            # Backward
            loss_T.backward()
            loss_E.backward()
            # Update weights
            optimizer_T.step()
            optimizer_E.step()

        else:

            with torch.no_grad():

                model_T.eval()
                model_E.eval()

                T_sum = 0
                E_sum = 0

                for x, y in self.dl['val']:
                    x = x.to(device)
                    y = y.to(device)
                    #forward pass
                    T = model_T(x)
                    E = model_E(x)
                    # Losses
                    val_loss_T = criterion(T, y[:,0].view(-1,1))
                    val_loss_E = criterion(E, y[:,1].view(-1,1))
                    val_loss_T_sum += val_loss_T/len(y[:,0])
                    val_loss_E_sum += val_loss_E/len(y[:,1])

                    # Mean Square Error
                    T_sum += torch.sum( torch.abs( T.to('cpu') - y[:,0].to('cpu') ) )
                    E_sum += torch.sum( torch.abs( E.to('cpu') - y[:,1].to('cpu') ) )

                # Times
                elapsed_time        = (float(time.time())-zero_time)
                time_per_epoch      = elapsed_time/(epoch+1)
                remaining_epochs    = num_epochs - epoch
                remaining_time      = time_per_epoch * remaining_epochs

                # Information
                train_loss_T.append(float(loss_T_sum.cpu().detach().numpy()))
                train_loss_E.append(float(loss_E_sum.cpu().detach().numpy()))
                validation_loss_T.append(float(val_loss_T_sum.cpu().detach().numpy()))
                validation_loss_E.append(float(val_loss_E_sum.cpu().detach().numpy()))

                print(f'Epoch: {epoch+1} of {num_epochs} || Remaining time: {time.strftime("%H:%M:%S",  time.gmtime(remaining_time))}')
                print(f' Training T loss: {loss_T_sum:.4f} || Validation T loss: {val_loss_T_sum:.4f} ')
                print(f' Training E loss: {loss_E_sum:.4f} || Validation E loss: {val_loss_E_sum:.4f} ')

    if representation:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(train_loss_T,label='train loss (T)')
        plt.plot(train_loss_E,label='train loss (E)')
        plt.plot(validation_loss_T,label='validation loss (T)')
        plt.plot(validation_loss_E,label='validation loss (E)')
        plt.legend()
        plt.grid()
        plt.show()


    self.model_T = model_T
    self.model_E = model_E
    return model_T, model_E

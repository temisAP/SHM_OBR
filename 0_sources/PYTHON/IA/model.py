import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Weights initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

# Compress sensor layer
class First_Stage_Layer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self):
        super().__init__()

        # Layer secuence
        self.FC = nn.Sequential(
            nn.Linear(12001, 6000),
            nn.Tanh(),
            nn.Linear(6000, 3000),
            nn.Tanh(),
            # nn.Dropout(p=0.2),
            nn.Linear(3000, 1500),
            nn.Tanh(),
            # nn.Dropout(p=0.2),
            nn.Linear(1500, 750),
            nn.Tanh(),
            # nn.Dropout(p=0.2),
            nn.Linear(750, 375)
        ).apply(init_weights)
    def forward(self, x):
        x = self.FC(x)
        return x

class Second_Stage_Layer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self):
        super().__init__()
        # Layer secuence
        self.FC = nn.Sequential(
            nn.Linear(375, 350),
            nn.Tanh(),
            nn.Linear(350, 300),
            nn.Tanh(),
            nn.Linear(300, 250),
            nn.Tanh(),
            nn.Linear(250, 200),
            nn.Tanh(),
            nn.Linear(200, 150),
            nn.Tanh(),
            nn.Linear(150, 100),
            nn.Tanh(),
            nn.Linear(100, 50),
            nn.Tanh(),
            nn.Linear(50, 25),
            nn.Tanh(),
            nn.Linear(25, 16),
            nn.Tanh(),
            nn.Linear(16, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
            nn.ReLU()
        ).apply(init_weights)
    def forward(self, x):
        x = self.FC(x)
        return x

class TE(nn.Module):
    def __init__(self):
        super(TE, self).__init__()
        self.cs0 = First_Stage_Layer()
        self.fc_T = Second_Stage_Layer()
        self.fc_E = Second_Stage_Layer()
    def forward(self, x):
        bs = x.shape[0]
        x0 = self.cs0(x.reshape(bs,-1))

        T = self.fc_T(x0).flatten()
        E = self.fc_E(x0).flatten()

        return T, E

class TE_single(nn.Module):
    def __init__(self,TE,n):
        super(TE_single, self).__init__()
        self.TE = TE
        self.n = n
        for p in self.TE.parameters():
            p.requires_grad=False

    def forward(self, x):
        y = self.TE(x)
        y = y[self.n]
        print(y)
        return y

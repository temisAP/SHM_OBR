import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Compress sensor layer
class First_Stage_Layer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, size_in, size_out):
        super().__init__()
        # Attribute init
        self.size_in, self.size_out = size_in, size_out
        # Layer secuence
        self.FC = nn.Sequential(
            nn.Linear(self.size_in, 1500),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(1500, 1000),
            nn.Tanh(),
            nn.Linear(1000, 750),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(750, 500),
            nn.Tanh(),
            # nn.Dropout(p=0.2),
            nn.Linear(500, 250),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(250, self.size_out)
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
            nn.Linear(125*8, 500),
            nn.Tanh(),
            nn.Linear(500, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 16),
            nn.Tanh(),
            # nn.Dropout(p=0.2),
            nn.Linear(16, 1),
            nn.ReLU()
        ).apply(init_weights)
    def forward(self, x):
        x = self.FC(x)
        return x

class splitter(nn.Module):
    """
        Class to joint Stages and regress
        temperature or deformation from signal
    """
    def __init__(self):
        super(splitter, self).__init__()
        self.cs0 = First_Stage_Layer(2000, 125)
        self.cs1 = First_Stage_Layer(2000, 125)
        self.cs2 = First_Stage_Layer(2000, 125)
        self.cs3 = First_Stage_Layer(2000, 125)
        self.cs4 = First_Stage_Layer(2000, 125)
        self.cs5 = First_Stage_Layer(2000, 125)
        self.cs6 = First_Stage_Layer(2000, 125)
        self.cs7 = First_Stage_Layer(2000, 125)

        self.fc = Second_Stage_Layer()

    def forward(self, x):

        bs = x.shape[0]
        x = x.reshape(bs,-1,8)

        x0 = self.cs0(x[:,:,0].reshape(bs,-1))
        x1 = self.cs1(x[:,:,1].reshape(bs,-1))
        x2 = self.cs2(x[:,:,2].reshape(bs,-1))
        x3 = self.cs3(x[:,:,3].reshape(bs,-1))
        x4 = self.cs4(x[:,:,4].reshape(bs,-1))
        x5 = self.cs5(x[:,:,5].reshape(bs,-1))
        x6 = self.cs6(x[:,:,6].reshape(bs,-1))
        x7 = self.cs7(x[:,:,7].reshape(bs,-1))
        x = torch.cat((x0, x1, x2, x3, x4, x5, x6, x7), 1)

        out = self.fc(x).flatten()

        return out

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

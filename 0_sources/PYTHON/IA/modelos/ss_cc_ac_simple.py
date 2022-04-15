import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Compressor(nn.Module):
    """ Compressor for signal  """
    def __init__(self,input_dim,output_dim):
        super().__init__()
        # Dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim
        N = 3
        hidden_dim = np.linspace(input_dim,output_dim,N+1,dtype = np.int32)[1:-1]
        # Layer secuence
        self.FC = nn.Sequential(
            nn.Linear(input_dim,hidden_dim[0]),
            nn.Tanh(),
            #nn.Dropout(p=0.2),
            nn.Linear(hidden_dim[0],hidden_dim[1]),
            nn.Tanh(),
            #nn.Dropout(p=0.2),
            nn.Linear(hidden_dim[1],output_dim),
            nn.Tanh()
            ).apply(init_weights)
    def forward(self, x):
        x = self.FC(x)
        return x

class Regressor(nn.Module):
    """ Regressor for one characteristic """
    def __init__(self,input_dim,output_dim):
        super().__init__()
        # Dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim
        N = 3
        hidden_dim = np.linspace(input_dim,output_dim,N+1,dtype = np.int32)[1:-1]
        # Layer secuence
        self.FC = nn.Sequential(
            nn.Linear(input_dim,hidden_dim[0]),
            nn.Tanh(),
            #nn.Dropout(p=0.2),
            nn.Linear(hidden_dim[0],hidden_dim[1]),
            nn.Tanh(),
            #nn.Dropout(p=0.2),
            nn.Linear(hidden_dim[1],output_dim),
            nn.Tanh()
            ).apply(init_weights)
    def forward(self, x):
        x = self.FC(x)
        return x


class splitter(nn.Module):
    """
        Class to joint Stages and regress
        temperature or deformation from signal
    """

    def __init__(self, components = ['ss','cc','ac'], Ls = [1,2000,400] , rhos = [1,100,2]):
        super(splitter, self).__init__()

        # Signal components
        self.components = dict.fromkeys(components)     # Name
        self.Ls         = Ls                            # Lenght
        self.rhos       = rhos                          # Compression rate

        # Compressor layers
        self.C_ss = Compressor(Ls[0],int(Ls[0]/rhos[0]))
        self.C_cc = Compressor(Ls[1],int(Ls[1]/rhos[1]))
        self.C_ac = Compressor(Ls[2],int(Ls[2]/rhos[2]))

        # Regressor layers
        input_size  = sum( [int(L/rho) for L,rho in zip(Ls,rhos)] ) #- int(Ls[1]/rhos[1])
        self.R   = Regressor( input_size , 1)

    def forward(self, x):
        """
        X components:

                X[0:0]       || X[0           :Ls[0]]             || -> Spectral shift
                X[1:2001]    || X[Ls[0]       :Ls[0]+Ls[1]]       || -> Cross correlation
                X[2001:2401] || X[Ls[0]+Ls[1] :Ls[0]+Ls[1]+Ls[2]] || -> Autocorrelation

        """


        # Batch size = x[bs,:]
        bs = x.shape[0]
        Ls = self.Ls

        # First_Stage_Layers
        x1 = x[: ,             :Ls[0]       ].reshape(bs,-1)
        x2 = x[: ,  Ls[0]      :Ls[0]+Ls[1] ].reshape(bs,-1)
        x3 = x[: , -Ls[2]      :            ].reshape(bs,-1)
        y1 = x1
        y2 = self.C_cc(x2)
        y3 = self.C_ac(x3)

        x = torch.cat((y1,y2,y3), 1)

        # Second_Stage_Layers
        out = self.R(x)

        return out

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

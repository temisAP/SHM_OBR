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
        N = 20
        dims = np.linspace(input_dim,output_dim,N+1,dtype = np.int32)
        # Layer secuence
        self.FC = nn.Sequential()
        for dim in dims:
            nn.Linear(dim[0],dim[1]),
            nn.Tanh(),
            #nn.Dropout(p=0.2),
            nn.Linear(dim[1],dim[2]),
            nn.Tanh(),
            #nn.Dropout(p=0.2),
            nn.Linear(dim[2],dim[3]),
            nn.Tanh()
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

        # Compressor layers
        self.C = Compressor(16,4)

        # Regressor layers
        self.R   = Compressor(4,1)

    def forward(self, x):
        # Batch size = x[bs,:]
        bs = x.shape[0]

        # First_Stage_Layer
        y = self.C(x)

        # Second_Stage_Layer
        out = self.R(y)

        return out

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

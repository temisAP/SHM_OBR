import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Linear(nn.Module):
    """ Linear compressor/expansor """
    def __init__(self,input_dim,output_dim,N):
        super().__init__()
        # Dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim
        dimensions = np.linspace(input_dim,output_dim,N+1,dtype = np.int32)
        # Layer secuence
        self.FC = nn.Sequential()
        for i in range(N):
            self.FC.add_module("Linear"+str(i),nn.Linear(dimensions[i],dimensions[i+1]))
            self.FC.add_module("BatchNorm"+str(i),nn.BatchNorm1d(dimensions[i+1]))
            self.FC.add_module("Tanh"+str(i),nn.Tanh())
        self.FC.add_module("Linear"+str(N),nn.Linear(dimensions[-1],output_dim))
    def forward(self,x):
        return self.FC(x)


class splitter(nn.Module):
    """
        Class to joint Stages and regress
        temperature or deformation from signal
    """

    def __init__(self):
        super(splitter, self).__init__()

        # Compressor layers
        self.C = Linear(16,1000,7)

        # Regressor layers
        self.R = Linear(1000,1,8)

    def forward(self, x):
        # Batch size = x[bs,:]
        bs = x.shape[0]

        # First_Stage_Layer
        y = self.C(x)

        # Second_Stage_Layer
        out = self.R(y)

        return out

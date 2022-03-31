import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Compressor(nn.Module):
    """ Compressor/expansor for raw atributes (ss,cc,ac) """
    def __init__(self,input_dim,output_dim):
        super().__init__()
        # Dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim
        N = 2
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

class Convolutional(nn.Module):
    """ Takes 3 channels and returns 1 using a CNN (convolutional neural network)  """
    def __init__(self,input_channels,output_channels,input_dim,output_dim):
        super().__init__()
        # Dimensions and channels
        self.input_channels  = input_channels
        self.output_channels = output_channels
        self.output_dim = int(output_dim)
        self.input_dim = int(input_dim)
        
        # Number of channels
        N = 2
        channels = np.linspace(input_channels,output_channels,N+1,dtype = np.int32)
        
        # Output dimension when convolutional cycle has finished
        padding = 1     # default 0
        dilation = 1    # default 1
        kernel_size = 3 # to be set
        stride = 1      # default 1 
        
        B = stride
        A = (2*padding-dilation*(kernel_size-1)-1) / stride + 1

        L_out = int(input_dim/B**N + sum([A/B**k for k in range(0,N)]))

        # Convolutional secuence
        self.CNN = nn.Sequential()
        for i in range(N):
            self.CNN.add_module("Conv"+str(i),nn.Conv1d(channels[i],channels[i+1],
                kernel_size = kernel_size,padding = padding, dilation=dilation, stride = stride))
            self.CNN.add_module("Tanh"+str(i),nn.Tanh())

        # Fully connected secuence
        self.FC = nn.Sequential()
        self.FC.add_module("Linear",nn.Linear(L_out,output_dim))
        self.FC.add_module("Sigmoid",nn.Sigmoid())

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        bs = x.shape[0]
        x   = self.CNN(x)
        out = self.FC(x.reshape(bs,-1))
        return out

class Regressor(nn.Module):
    """ Regressor for one characteristic, uses a fully conected stages """
    def __init__(self,input_dim,output_dim):
        super().__init__()
        # Dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim
        N = 4
        dimensions = np.linspace(input_dim,output_dim,N+1,dtype = np.int32)
        # Layer secuence
        self.FC = nn.Sequential()
        for i in range(N):
            self.FC.add_module("Linear"+str(i),nn.Linear(dimensions[i],dimensions[i+1]))
            self.FC.add_module("BatchNorm"+str(i),nn.BatchNorm1d(dimensions[i+1]))
            self.FC.add_module("Tanh"+str(i),nn.Tanh())
        self.FC.add_module("Linear"+str(N),nn.Linear(dimensions[-1],output_dim))
        self.FC.add_module("Softmax"+str(N),nn.Softmax())
    def forward(self,x):
        return self.FC(x)

class splitter(nn.Module):
    """
        Class to joint Stages and regress
        temperature or deformation from signal
    """

    def __init__(self,  Ls = [1,2000,400] , dims = [100,500]):
        super(splitter, self).__init__()

        # Signal lenghts
        self.Ls         = Ls

        # Compressor layers
        self.C_ss = Compressor(Ls[0],dims[0])
        self.C_cc = Compressor(Ls[1],dims[0])
        self.C_ac = Compressor(Ls[2],dims[0])

        # Convolutional layer
        self.CNN = Convolutional(3,1,dims[0],dims[1])

        # Regressor layer
        self.R   = Regressor( dims[1] , 1)

    def forward(self, x):
        """
        X components:

                X[0:0]       || X[0           :Ls[0]]             || -> Spectral shift
                X[1:2001]    || X[Ls[0]       :Ls[0]+Ls[1]]       || -> Cross correlation
                X[2001:2401] || X[Ls[0]+Ls[1] :Ls[0]+Ls[1]+Ls[2]] || -> Autocorrelation

        """

        # Compressor layers
        bs = x.shape[0] # Batch size = x[bs,:] 
        Ls = self.Ls

        x1 = x[: ,             :Ls[0]       ].reshape(bs,-1)
        x2 = x[: ,  Ls[0]      :Ls[0]+Ls[1] ].reshape(bs,-1)
        x3 = x[: , -Ls[2]      :            ].reshape(bs,-1)

        y1 = self.C_ss(x1)
        y2 = self.C_cc(x2)
        y3 = self.C_ac(x3)

    
        # Convolutional layer
        x = torch.cat([y1,y2,y3], dim= 1).reshape(bs,3,-1)
        x = self.CNN(x)

        # Regressor layer
        out = self.R(x)

        return out

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

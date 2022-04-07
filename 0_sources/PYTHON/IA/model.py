import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Filter(nn.Module):
    """ Filter inputs to get a smother signal """
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
        self.FC.add_module("Tanh"+str(N),nn.Tanh())
    def forward(self,x):
        return self.FC(x)

class Convolutional(nn.Module):
    """ Takes 4 channels and returns 1 using a CNN (convolutional neural network)  """
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
        N = 3
        dimensions = np.linspace(input_dim,output_dim,N+1,dtype = np.int32)
        # Layer secuence
        self.FC = nn.Sequential()
        for i in range(N):
            self.FC.add_module("Linear"+str(i),nn.Linear(dimensions[i],dimensions[i+1]))
            self.FC.add_module("BatchNorm"+str(i),nn.BatchNorm1d(dimensions[i+1]))
            self.FC.add_module("Tanh"+str(i),nn.Tanh())
        self.FC.add_module("Linear"+str(N),nn.Linear(dimensions[-1],output_dim))
        self.FC.add_module("Tanh"+str(N),nn.Tanh())
    def forward(self,x):
        return self.FC(x)

class splitter(nn.Module):
    """
        Class to joint Stages and regress
        temperature or deformation from signal
    """

    def __init__(self,  dims = [2000,500,250,1]):
        super(splitter, self).__init__()

        self.dims = dims
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Filter/transformation layers
        self.Filter = Filter(dims[0],dims[1])

        # Convolutional layers
        self.Convolution = Convolutional(6,1,dims[1],dims[2])

        # Regressor layers
        self.Regression   = Regressor(dims[2]+1,dims[3])

    def forward(self, x):

        # Reshape
        bs = x.shape[0] # Batch size = x[bs,:]
        freqs = x[:,-1:]            # Frequencies
        x     = x[:,:-1]            # Correlations
        x     = x.reshape(bs,6,-1)  # Reshape as a tensor whose rows are the six possible correlations

        # Filter/transformation of each signal
        y = torch.from_numpy( np.empty(( bs,6,self.dims[1] )) ).float().to(self.device)
        for i in range(6):
            y[:,i,:] = self.Filter(x[:,i,:])

        # Convolutional layers
        y = y.reshape(bs,6,-1)
        y = self.Convolution(y)

        # Regressor layer
        y = torch.cat((y,freqs), 1)
        out = self.Regression(y)

        return out

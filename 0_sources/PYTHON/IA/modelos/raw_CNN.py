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
        N = 1
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
    """ Takes 4 channels and returns 1 using a CNN (convolutional neural network)  """
    def __init__(self,input_channels,output_channels,input_dim,output_dim):
        super().__init__()
        # Dimensions and channels
        self.input_channels  = input_channels
        self.output_channels = output_channels
        self.output_dim = int(output_dim)
        self.input_dim = int(input_dim)

        # Number of channels
        N = 1
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

class splitter(nn.Module):
    """
        Class to joint Stages and regress
        temperature or deformation from signal
    """

    def __init__(self,  dims = [2000,500,250,150,1]):
        super(splitter, self).__init__()

        self.dims = dims
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Filter/transformation layers
        self.Filter = Filter(dims[0],dims[1])

        # Convolutional layers
        self.Convolution = Convolutional(4,1,dims[1],dims[2])

        # Interpretation of the signals
        self.Interpretation = Convolutional(2,1,dims[2],dims[3])

        # Regressor layers
        self.Regression   = Regressor(dims[3],dims[4])

    def forward(self, x):
        """
        X components:

            X = [P1.real, P1.imag,
                S1.real, S1.imag,
                P2.real, P2.imag,
                S2.real, S2.imag]

        """

        # Reshape
        bs = x.shape[0] # Batch size = x[bs,:]

        x = x.reshape(bs,8,-1)
        y = torch.from_numpy( np.empty(( bs,8,self.dims[1] )) ).float().to(self.device)

        # Filter/transformation of each signal
        for i in range(8):
            y[:,i,:] = self.Filter(x[:,i,:])

        # Convolutional layers
        Ps = torch.cat([y[:,0,:],y[:,1,:],y[:,4,:],y[:,5,:]], dim= 1).reshape(bs,4,-1)
        Ss = torch.cat([y[:,2,:],y[:,3,:],y[:,6,:],y[:,7,:]], dim= 1).reshape(bs,4,-1)

        P_conv = self.Convolution(Ps)
        S_conv = self.Convolution(Ss)

        x = torch.cat([P_conv, S_conv], dim= 1).reshape(bs,2,-1)

        # Interpetation from two signals
        x = self.Interpretation(x)

        # Regressor layer
        out = self.Regression(x)

        return out

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np

#from apex.parallel import SyncBatchNorm

class new_Module(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.prob = None
        self.var = 1
        
        self.__name__ = "new_Module"


def Set_prob(net, val):
    if hasattr(net, 'prob'):
        net.prob = val
    for module in net.children():
        Set_prob(module, val)

def Set_var(net, val):
    if hasattr(net, 'var'):
        net.var = val
    for module in net.children():
        Set_var(module, val)

def gaussian_augment(x, prob=None, var=1):
    if prob is not None:
        #matrix = torch.where(torch.rand(x.shape) < prob, torch.normal(1, var, x.shape), torch.ones(x.shape)).to(x.device)
        normal = torch.from_numpy(np.random.normal(1, var, x.shape)).float()
        matrix = torch.where(torch.rand(x.shape) < prob, normal, torch.ones(x.shape)).to(x.device)
        x = matrix * x
        
    return x


class aug_Conv2d(new_Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
    
    def forward(self, x):
        weight = gaussian_augment(self.weight, self.prob, self.var)
        
        if self.bias is not None:
            bias = gaussian_augment(self.bias, self.prob, self.var)
            x = F.conv2d(x, weight, bias=bias, stride=self.stride, padding=self.padding, dilation=self.dilation)
        else:
            x = F.conv2d(x, weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation)
        
        return x

class aug_Linear(new_Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None    
    
    def forward(self, x):
        weight = gaussian_augment(self.weight, self.prob, self.var)
        
        if self.bias is not None:
            bias = gaussian_augment(self.bias, self.prob, self.var)
            x = F.linear(x, weight, bias=bias)
        else:
            x = F.linear(x, weight, bias=None)
        
        return x

class aug_BatchNorm2d(new_Module):
    def __init__(self, channels, eps=1e-05, momentum=0.1, affine=False):
        super().__init__()
        
        self.weight = nn.Parameter(torch.ones(1,channels,1,1))
        self.bias = nn.Parameter(torch.zeros(1,channels,1,1))
        
        self.bn = nn.BatchNorm2d(channels, eps=eps, momentum=momentum/2, affine=False)
        #self.bn = SyncBatchNorm(channels, eps=eps, momentum=momentum/2, affine=False)
        
    def forward(self, x):
        weight = gaussian_augment(self.weight, self.prob, self.var)
        bias = gaussian_augment(self.bias, self.prob, self.var)
        
        x = weight * self.bn(x) + bias
        
        return x


    
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    a = torch.randn((1,3,256,256)).to(device)
    b = aug_Conv2d(3, 1, 3).to(device)
    
    Set_prob(b, None)
    c = b(a)
    print(c.var())
    
    Set_prob(b, 0.05)
    c = b(a)
    print(c.var())

    Set_prob(b, 0.2)
    c = b(a)
    print(c.var())
    
    